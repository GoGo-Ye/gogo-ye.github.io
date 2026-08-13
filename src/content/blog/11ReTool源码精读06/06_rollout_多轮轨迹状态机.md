---
title: ReTool 源码精读06
publishDate: 2026-08-13
description: '第六章：`rollout.py`——多轮轨迹状态机'
tags:
  - 笔记
---

# 第六章：`rollout.py`——多轮轨迹状态机

源码位置：`https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/05-retool/rollout.py`

## 6.1 先抓住整章主线

`rollout.py` 是前五章第一次真正汇合的地方：

```text
data.py：MathExample
    ↓
protocol.py：初始 prompt / 工具协议 / token 连续续接
    ↓
rollout.py 状态机
    ├── 请求模型采样 assistant action
    ├── 解析 answer / tool / invalid
    ├── 调用 sandbox.py 执行 Python
    ├── 把 observation 接回下一轮 prompt
    ├── 控制轮数、工具次数和 token 预算
    └── 循环直到所有轨迹结束
    ↓
reward.py：每条轨迹得到 ±1
    ↓
同题组中心化：每条轨迹得到 advantage
    ↓
train.py：Trajectory.turns → PPO Datum
```

一句话概括：

> rollout 把“模型生成—环境执行—观察反馈—模型续写”驱动成多轮轨迹，并保留训练所需的真实 token、旧 logprob、最终 reward 和组内 advantage。

## 6.2 什么是 code-interlaced trajectory

普通单轮生成可以写成：

```text
prompt → assistant final answer
```

ReTool 的代码交织轨迹是：

```text
prompt
  → assistant 推理 + code tool call
  → tool observation
  → assistant 继续推理 + code tool call
  → tool observation
  → assistant final answer
```

“interlaced”强调两类 token 交替出现：

```text
模型动作 token：assistant completion
环境反馈 token：tool observation
```

工具调用标签和其中的 Python 代码是 assistant 自己生成的，所以属于模型动作；Python stdout、traceback 和 timeout 信息由环境生成，所以属于 observation。

## 6.3 `round`、`assistant turn` 和 `training step`

三个概念不要混淆。

### assistant turn

某一条轨迹的一次模型生成。每调用一次 sampler 并消费一个 sequence，该轨迹就增加一个 `AssistantTurn`。

### rollout round

状态机的一轮同步推进：给当前所有活跃轨迹各采样一次，并并发执行这一轮出现的工具调用。不同轨迹可能在同一 round 结束或继续。

### training step

训练脚本选出一批问题，生成所有分组轨迹，计算 reward/advantage，构造 Datum，执行一次逻辑上的 PPO 参数更新。

关系大致是：

```text
1 training step
  └── 1 rollout batch
        ├── first round
        ├── later round 1
        ├── later round 2
        └── ...
              └── 每条活跃轨迹各产生 1 assistant turn
```

## 6.4 `RolloutConfig`：三类控制参数

```python
@dataclass(frozen=True)
class RolloutConfig:
    group_size = 8
    max_code_calls = 4
    max_assistant_turns = 6
    max_trajectory_tokens = 8192
    max_assistant_tokens = 1024
    max_tool_response_tokens = 512
    temperature = 1.0
    top_p = 1.0
    seed = 42
```

### 分组参数

`group_size`：同一道问题首轮采样多少条候选轨迹，用于之后的同题相对比较。

### 轨迹预算参数

| 参数 | 限制对象 |
| --- | --- |
| `max_code_calls` | 一条轨迹成功接入历史的工具调用次数 |
| `max_assistant_turns` | 一条轨迹最多记录多少次 assistant 生成 |
| `max_trajectory_tokens` | 当前完整 prompt/轨迹的 token 总预算 |
| `max_assistant_tokens` | 单个 assistant turn 的最大生成 token |
| `max_tool_response_tokens` | 单次 tool content 的最大 token |

### 采样参数

`temperature`、`top_p` 和 `seed` 控制采样随机性。

配置对象使用 `frozen=True`，整个 rollout 过程中只读取，不应被某条轨迹修改。

## 6.5 四个核心对象先总览

```text
MathExample
    ↓ 创建
Trajectory
    ↓ make_request()
SampleRequest
    ↓ sampler response
AssistantTurn 写入 Trajectory.turns
    ↓ 如果是合法工具调用
PendingExecution
    ↓ sandbox result
更新 Trajectory，继续或结束
```

| 对象 | 生命周期 | 核心职责 |
| --- | --- | --- |
| `Trajectory` | 整条轨迹 | 保存完整动态状态和训练结果 |
| `SampleRequest` | 一次模型请求 | 描述 prompt、采样数、预算、seed 和写回位置 |
| `AssistantTurn` | 一次模型动作 | 保存本轮真实 prompt/completion/logprob/text |
| `PendingExecution` | 一次待执行工具调用 | 把代码、轨迹引用和续接上下文带到沙箱结果写回阶段 |

## 6.6 `AssistantTurn`：训练最关心的动作记录

```python
@dataclass
class AssistantTurn:
    prompt_tokens: list[int]
    completion_tokens: list[int]
    logprobs: list[float]
    text: str
```

字段边界：

```text
prompt_tokens     → 本轮生成前的完整模型输入 Pᵢ
completion_tokens → sampler 真实返回的模型动作 Cᵢ
logprobs          → 旧策略对 Cᵢ 每个 token 的 logprob Lᵢ
text              → 可读文本，用于协议解析、代码提取和日志
```

最重要的不变量：

```text
len(completion_tokens) == len(logprobs)
```

`completion_tokens` 与 `logprobs` 是 PPO 训练事实；`text` 是语义解释。即使本轮输出格式非法或最终 reward 为负，实际采样过的动作仍会先写进 `turns`。

## 6.7 `Trajectory`：一条轨迹的完整状态

```python
@dataclass
class Trajectory:
    example: MathExample
    group_index: int
    messages: list[dict]
    next_prompt_tokens: list[int] | None = None
    question_index: int = 0
    turns: list[AssistantTurn] = ...
    code_calls: int = 0
    final_text: str = ""
    reward: float = -1.0
    advantage: float = 0.0
    valid_format: bool = False
    correct: bool = False
    done: bool = False
```

可以按四组理解。

### 静态归属

```text
example        → 对应的题目和参考答案
question_index → 题目在当前 rollout batch 中的位置
group_index    → 当前轨迹在同题 group 中的分支编号
```

### 双轨历史

```text
messages           → 可读的 system/user/assistant/tool 历史
next_prompt_tokens → 工具返回后连续构造的真实下一轮 token prompt
turns              → 每一次真实 assistant 动作
```

### 状态计数

```text
code_calls → 已成功把 tool observation 接入对话的次数
done       → 是否停止继续采样
```

### 终局训练信号

```text
final_text   → 交给 reward 的最后 assistant 文本
reward       → ±1 结果奖励
valid_format → 是否找到完整 boxed
correct      → 数学答案是否正确
advantage    → reward 减同题组均值
```

`Trajectory` 必须可变，因为 rollout 会逐轮原地更新它。

## 6.8 三种 index 分别是什么

假设一个 batch 有两道题，每题 `group_size=3`：

```text
question_index=0
├── group_index=0 → flattened trajectory_index=0
├── group_index=1 → flattened trajectory_index=1
└── group_index=2 → flattened trajectory_index=2

question_index=1
├── group_index=0 → flattened trajectory_index=3
├── group_index=1 → flattened trajectory_index=4
└── group_index=2 → flattened trajectory_index=5
```

| 名称 | 是否长期稳定 | 用途 |
| --- | --- | --- |
| `question_index` | 在本次 rollout 内稳定 | reward 分组、call ID |
| `group_index` | 在同题组内稳定 | 标识候选分支、call ID |
| `SampleRequest.trajectory_index` | 取决于当前索引的列表 | 把响应写回 root 或扁平轨迹 |

首轮 `trajectory_index` 指向 `roots`；后续则指向 `trajectories`。因此它不是轨迹永久 ID。

工具调用 ID：

```python
code-{question_index}-{group_index}-{code_calls + 1}
```

例如 `code-1-2-3` 表示当前 batch 第 2 道题、组内第 3 条轨迹的第 3 次工具调用。

## 6.9 `SampleRequest`：一次采样任务

```python
@dataclass(frozen=True)
class SampleRequest:
    trajectory_index: int
    prompt_tokens: list[int]
    num_samples: int
    max_tokens: int
    seed: int
```

其字段回答五个问题：

```text
结果写回哪条轨迹？       → trajectory_index
模型看到什么 token？      → prompt_tokens
共享 prompt 采多少候选？ → num_samples
每个候选最多生成多少？   → max_tokens
本次随机种子是多少？      → seed
```

首轮每道题一个请求，`num_samples=group_size`；后续每条未结束轨迹一个请求，`num_samples=1`。

## 6.10 `PendingExecution`：跨阶段保存工具调用

```python
@dataclass(frozen=True)
class PendingExecution:
    trajectory: Trajectory
    code: str
    call_id: str
    messages_before_assistant: list[dict]
    assistant_text: str
    prompt_tokens: list[int]
    completion_tokens: list[int]
```

它在“assistant 已生成工具调用”和“沙箱结果尚未写回”之间充当事务记录：

```text
trajectory                 → 结果写回哪个可变对象
code                       → 沙箱执行什么
messages_before_assistant  → 当前 assistant 生成前的消息快照
assistant_text             → observation 无法接入时作为 final_text
prompt_tokens + completion → 连续拼接下一轮真实 token
```

它不重复保存 `logprobs`，因为 logprob 已经连同 completion 写入 `AssistantTurn`。

## 6.11 顶层状态机鸟瞰

`rollout_batch_async()` 的完整阶段是：

```text
1. 每题创建一个 root Trajectory
2. 每题构造一个首轮 SampleRequest
3. 并发采样所有题的首轮 group
4. 每个 sequence 深拷贝 root，形成独立分支
5. 解析首轮 assistant 输出
6. 并发执行首轮所有工具调用
7. while 仍有未结束轨迹：
     a. 为每条活跃轨迹构造单样本请求
     b. 并发采样
     c. 解析全部输出
     d. 并发执行其中的工具调用
     e. 写回 observation 或终止
8. 所有轨迹逐条计算 reward
9. 按 question_index 计算组内 advantage
10. 返回 list[Trajectory]
```

## 6.12 每道题先创建一个 root

```python
roots = [
    Trajectory(
        example=example,
        group_index=0,
        messages=initial_messages(example.question),
        question_index=question_index,
    )
    for question_index, example in enumerate(examples)
]
```

此时一题只有一个 root：

```text
Trajectory
├── messages = [system, user]
├── turns = []
├── code_calls = 0
├── done = False
└── reward/advantage 尚未计算
```

`group_index=0` 只是临时值，真正分叉后每个 branch 会重新赋成 `0 ... group_size-1`。

## 6.13 `make_request()`：prompt 与生成预算

prompt 有两条来源：

```python
prompt_tokens = (
    trajectory.next_prompt_tokens
    if trajectory.next_prompt_tokens is not None
    else build_prompt(tokenizer, trajectory.messages)
)
```

首轮从 `[system, user]` 完整渲染；工具调用后的后续轮使用连续构造好的真实 token，不重渲染历史。

本轮可生成上限：

```python
remaining = max_trajectory_tokens - len(prompt_tokens)
max_tokens = min(max_assistant_tokens, remaining)
```

假设总预算 8192、单轮上限 1024：

| 当前 prompt 长度 | 剩余总预算 | 本轮 `max_tokens` |
| ---: | ---: | ---: |
| 5000 | 3192 | 1024 |
| 7800 | 392 | 392 |
| 8192 | 0 | 不采样，直接 done |

若 `max_tokens <= 0`，函数设置 `trajectory.done=True` 并返回 `None`。这里没有截断旧历史，也没有滑动窗口。

## 6.14 为什么首轮一次采样整个 group

首轮每道题只有一个共享 prompt，因此只创建一个请求：

```python
request = make_request(
    tokenizer,
    root,
    trajectory_index=question_index,
    num_samples=config.group_size,
    seed=config.seed + question_index,
    config=config,
)
```

若有 2 道题、`group_size=3`：

```text
请求 Q0：prompt of question 0，num_samples=3
请求 Q1：prompt of question 1，num_samples=3
```

这比为相同 prompt 重复发 3 个单样本请求更自然，也明确表达“同题一组候选”。不同题的两个请求通过 `asyncio.gather()` 并发。

采样服务必须返回精确数量：

```python
if len(response.sequences) != group_size:
    raise ValueError(...)
```

## 6.15 root 如何分叉为独立 branch

对首轮返回的每个 sequence：

```python
branch = copy.deepcopy(root)
branch.group_index = group_index
trajectories.append(branch)
begin_advance(branch, ..., sequence, ...)
```

深拷贝很重要。否则多个分支可能共享同一个 `messages` 或 `turns` 列表：一条轨迹追加 tool observation，会污染同题其他候选。

分叉后：

```text
一个 root prompt
├── branch 0：独立 messages / turns / code_calls / done
├── branch 1：独立 messages / turns / code_calls / done
└── branch 2：独立 messages / turns / code_calls / done
```

`MathExample` 本身是 frozen 的字符串值对象，共享或复制都不会被轨迹修改。

## 6.16 为什么后续每条轨迹只能单独采样

首轮之后，各分支可能：

- 直接回答并结束；
- 调用不同 Python 代码；
- 得到不同 stdout 或 traceback；
- 因输出长度不同而拥有不同 token prompt。

因此它们不再共享 prompt：

```text
branch 0 prompt = 原 prompt + code A + observation A
branch 1 prompt = 原 prompt + code B + observation B
branch 2 已结束
```

后续循环为每条活跃轨迹创建 `num_samples=1` 的请求。虽然请求各自独立，但多个 `sample_async` 仍在同一个 round 中并发调度。

## 6.17 `sample_requests_async()`：并发采样与顺序映射

每个 request 转为：

```python
SamplingParams(
    max_tokens=request.max_tokens,
    seed=request.seed,
    stop=stop_sequences(tokenizer),
    temperature=config.temperature,
    top_p=config.top_p,
)
```

prompt 则转成：

```python
trio.ModelInput.from_ints(request.prompt_tokens)
```

所有 awaitable 交给：

```python
await asyncio.gather(*tasks)
```

并发任务完成顺序可能不同，但 `gather()` 的返回列表保持输入任务顺序。因此：

```text
requests[i] ↔ responses[i]
```

后面再用 `zip(..., strict=True)`，防止列表长度不一致时静默截断。

源码没有在这一层添加 retry、单独 timeout 或并发 semaphore；相关服务限制交给 PyTRIO 客户端。任何采样基础设施异常会让整个 gather 向上抛出。

## 6.18 `read_sequence()`：守住 token/logprob 对齐

```python
tokens = [int(token) for token in sequence.tokens]
logprobs = [float(value) for value in sequence.logprobs]
```

随后立即检查：

```python
if len(tokens) != len(logprobs):
    raise ValueError(...)
```

文本优先使用服务返回的 `sequence.text`；只有它是 `None` 时，才从 token 解码：

```python
tokenizer.decode(tokens, skip_special_tokens=True)
```

源码信任服务保证 `sequence.text` 与 `sequence.tokens` 表达同一输出，没有额外逐 token 一致性校验。

这个边界很重要：

```text
协议解析和代码执行使用 text
PPO 训练使用 tokens + logprobs
```

## 6.19 `begin_advance()`：一轮输出的核心分岔

执行顺序：

```text
1. read_sequence() 取得 tokens/logprobs/text
2. text.strip()
3. 立即追加 AssistantTurn
4. parse_assistant(text)
5. 决定执行工具还是结束轨迹
```

“先记录再解析”意味着所有实际采样动作都会留在 `turns`，包括正确答案、错误答案、非法工具格式，以及达到次数或轮数上限时生成的最后动作。

## 6.20 `can_code` 的三个必要条件

```python
can_code = (
    parsed.kind == "tool"
    and trajectory.code_calls < config.max_code_calls
    and len(trajectory.turns) < config.max_assistant_turns
)
```

只有三项同时满足，代码才会执行并继续轨迹。

注意当前 turn 已经 append，因此 `len(turns)` 包含本轮。默认最多 6 轮时：

```text
第 1～5 轮可以调用工具并等待下一轮
第 6 轮即使生成合法 tool call，也不会执行
```

这样避免第 6 轮调用工具后还必须出现第 7 轮才能给最终答案。

## 6.21 不继续工具时的统一终止分支

如果 `can_code=False`：

```python
trajectory.messages.append({"role": "assistant", "content": text})
trajectory.final_text = text
trajectory.done = True
return None
```

以下情况都走这里：

| 当前输出/状态 | 结果 |
| --- | --- |
| 普通最终回答 | 正常结束，交给 reward |
| 非法工具格式 | 结束，原文本交给 reward |
| 工具次数已满又生成工具调用 | 不执行该代码，工具调用文本成为 final_text |
| 已是最后允许轮次又生成工具调用 | 不执行该代码，工具调用文本成为 final_text |

reward 只看 `final_text` 中的 boxed，不会自动知道是哪一种终止原因。

## 6.22 合法工具调用如何形成 `PendingExecution`

若可以执行工具：

```text
生成 call_id
→ 保存当前 assistant 之前的 messages 快照
→ 把真实 assistant 文本追加到 trajectory.messages
→ 返回 PendingExecution
```

`messages_before_assistant` 必须在追加当前 assistant 前保存，因为 `build_next_prompt()` 使用占位 assistant 来推导 closing 和 observation 增量。

此时轨迹尚未增加 `code_calls`；只有沙箱结果成功接入历史后才增加。

## 6.23 同一 round 的工具调用如何并发

状态机先解析本 round 所有响应，把合法工具调用收集为：

```python
pendings: list[PendingExecution]
```

然后一次性执行：

```python
await asyncio.gather(
    *(sandbox.arun_code(p.code) for p in pendings)
)
```

每个 `arun_code()` 再通过线程池运行同步子进程，sandbox 内部的 `BoundedSemaphore` 限制实际同时执行的进程数量。

结果列表保持 pending 输入顺序，随后严格 zip 写回。每个 pending 又直接持有目标 `Trajectory` 引用，所以不会因异步完成顺序把 observation 写错轨迹。

## 6.24 并发是 round 内并发，round 间有屏障

一轮的时序是：

```text
并发采样所有活跃轨迹
        ↓ 等全部采样完成
解析全部响应
        ↓
并发执行所有 tool calls
        ↓ 等全部工具完成
统一写回结果
        ↓
进入下一 round
```

采样阶段和工具执行阶段不会重叠，下一 round 也必须等待当前 round 最慢的任务。因此它是“round 内并发、round 间同步屏障”，不是每条轨迹完全独立地一路跑到底。

好处是状态和进度容易统一管理；代价是长尾工具调用会拖慢整个 round。

## 6.25 `fit_tool_content()`：同时满足两类预算

沙箱先把结果格式化成 `content`。接下来必须同时满足：

```text
条件 1：token_count(content) <= max_tool_response_tokens
条件 2：len(next_prompt) <= max_trajectory_tokens
```

只有条件 1 通过，才用真实模板构造下一轮 prompt；这样总预算检查包含：

- 当前真实 prompt；
- 当前真实 completion；
- 缺失 assistant closing；
- tool role 和内容；
- tool closing；
- 下一轮 assistant generation prefix。

若任一条件不满足，就保留 tool content 尾部约 70%：

```python
content = "[... truncated ...]\n" + content[-int(len(content) * 0.7):]
```

这是按字符迭代缩短，不是一次精确截到 N 个 token。保留尾部是因为计算结果和 traceback 关键信息通常位于末尾。

若内容已不超过 64 个字符仍放不下，返回 `None`，让轨迹终止。

## 6.26 `finish_advance()`：observation 成功或失败写回

### 成功装入预算

```python
trajectory.messages.append(tool_message(...))
trajectory.next_prompt_tokens = next_prompt_tokens
trajectory.code_calls += 1
```

此时 `done` 仍为 False，下一 round 继续采样。

### 无法装入预算

```python
trajectory.final_text = pending.assistant_text
trajectory.done = True
```

此时：

- 不追加 tool message；
- 不增加 `code_calls`；
- 但沙箱代码事实上已经执行过；
- 当前 assistant 工具调用文本被当作 final_text 评分。

所以 `code_calls` 更准确地表示“成功接入对话历史的工具调用数”，不一定等于真实沙箱执行次数。

## 6.27 一条两轮轨迹的完整状态变化

假设模型第一轮调用工具，第二轮回答。

### 初始状态

```text
messages = [system, user]
next_prompt_tokens = None
turns = []
code_calls = 0
done = False
```

### 构造首轮请求

```text
P₁ = build_prompt(messages)
SampleRequest(prompt=P₁, num_samples=group_size, ...)
```

### 第一轮 sampler 返回

```text
C₁ = 推理 + tool_call 的真实 token
L₁ = 每个 C₁ token 的旧 logprob
text₁ = 可读工具调用文本
```

状态先更新：

```text
turns = [AssistantTurn(P₁, C₁, L₁, text₁)]
messages = [system, user, assistant tool_call]
```

创建 `PendingExecution`，并发执行代码。

### 沙箱返回 observation

```text
content₁ = stdout / stderr / timeout
D₁ = assistant closing 补全 + tool observation + 下一 assistant 前缀
P₂ = P₁ + C₁ + D₁
```

状态：

```text
messages = [system, user, assistant tool_call, tool observation]
next_prompt_tokens = P₂
code_calls = 1
done = False
```

### 第二轮 sampler 返回最终答案

```text
C₂ = 最终推理 + \boxed{answer}
L₂ = 与 C₂ 一一对应的旧 logprob
```

状态：

```text
turns = [
    AssistantTurn(P₁, C₁, L₁, text₁),
    AssistantTurn(P₂, C₂, L₂, text₂),
]
messages += assistant final answer
final_text = text₂
done = True
```

最后 `score_trajectory()` 写入 reward/correct/valid_format，再写 advantage。

## 6.28 这条轨迹的 token 不变量

记：

```text
Pᵢ：第 i 轮 prompt
Cᵢ：第 i 轮 completion
Lᵢ：第 i 轮旧 logprobs
Dᵢ：工具调用后的环境/结构增量
```

必须满足：

```text
len(Cᵢ) = len(Lᵢ)
Pᵢ₊₁ = Pᵢ + Cᵢ + Dᵢ
```

两轮轨迹最终序列：

```text
P₁ + C₁ + D₁ + C₂
```

训练 mask 预览：

```text
P₁ → context，advantage 0
C₁ → assistant action，trajectory advantage
D₁ → closing/tool observation/下一轮前缀，advantage 0
C₂ → assistant action，trajectory advantage
```

这就是为什么 `AssistantTurn` 必须保存每轮的完整 prompt，而不是只保存本轮新 observation。

## 6.29 三层前缀对齐保护

### 构造层

`build_next_prompt()` 直接返回：

```text
previous prompt + real completion + structural/observation delta
```

所以新 prompt 必然以前一轮 `Pᵢ+Cᵢ` 开头。

### chat template 层

`protocol.py` 用占位 assistant 检查模板能否稳定切出 closing 和 tool observation 增量。模板若改写历史会立即报错。

### 训练层

`train.py/build_datum()` 再检查：

```python
turn.prompt_tokens[:len(full_tokens)] == full_tokens
```

若下一轮 prompt 不是已有 token 轨迹的前缀扩展，就拒绝构造 PPO Datum。

因此是：

```text
rollout 连续构造
→ protocol 检查模板可切分
→ train 再检查真实前缀
```

## 6.30 `advance_round()`：后续 round 的批量推进器

它分两遍处理。

第一遍：

```text
requests/responses 严格配对
→ 找到 trajectory_index 对应轨迹
→ begin_advance()
→ 结束的立即更新 progress
→ 工具调用收集为 pendings
```

第二遍：

```text
并发执行全部 pendings
→ finish_advance() 逐条写回
→ 因预算失败而结束的更新 progress
```

仍能继续的轨迹不更新完成进度，留到后面 round。

## 6.31 while 循环如何收敛

循环条件：

```python
while any(not trajectory.done for trajectory in trajectories):
```

每轮只为 `done=False` 的轨迹建请求。轨迹最终会因以下原因之一结束：

1. 生成普通 answer；
2. 生成 invalid 输出；
3. 达到工具次数上限后仍生成 tool call；
4. 达到 assistant turn 上限后仍生成 tool call；
5. 当前 prompt 已无生成 token 预算；
6. tool observation 即使截断也放不进预算。

只要配置为正常正值并且采样/工具调用能返回，轮数和 token 上限共同保证轨迹不会无限续写。

## 6.32 同步入口与异步入口

训练代码处于同步循环，调用：

```python
rollout_batch(...)
```

它内部执行：

```python
asyncio.run(rollout_batch_async(...))
```

每次调用会创建并在结束后关闭一个 event loop；同一次 rollout 的所有 round 共用这个 loop。

若调用方已经位于运行中的 event loop，例如 `eval.py`，不能嵌套 `asyncio.run()`，应直接：

```python
await rollout_batch_async(...)
```

否则会出现 `asyncio.run() cannot be called from a running event loop`。

## 6.33 seed 如何分配

### 首轮

```text
seed = base_seed + question_index
```

同一道题的整个 group 使用一个多样本请求；各分支没有分别传独立 seed，组内随机流由 sampler 的 `num_samples` 语义管理。

### 后续轮

```text
seed = base_seed
     + flattened trajectory_index
     + len(turns) × 10000
```

首次后续采样时已经有 1 个 turn，因此偏移是 10000；再下一轮是 20000。

seed 是位置驱动的，不使用 `example.id`，也不包含 training step。每个 rollout batch 的 index 会从 0 重置。固定 seed 提高可复现性，但端到端结果还依赖采样后端、模型权重、prompt、并发动态 batching 和沙箱代码的确定性。

`10000` 只是人工分隔轮次的间隔，不是数学上严格无碰撞的编码；默认轨迹规模远小于 10000。

## 6.34 所有轨迹结束后才评分

主循环结束后：

```python
for trajectory in trajectories:
    score_trajectory(trajectory)
assign_group_advantages(trajectories)
```

`score_trajectory()` 调用：

```python
score_answer(
    trajectory.final_text,
    trajectory.example.answer,
)
```

并写回：

```text
reward
valid_format
correct
```

工具调用过程中不产生即时 reward；整条轨迹只在终点得到一次结果奖励。

## 6.35 advantage 为什么按 `question_index` 分组

```python
groups.setdefault(trajectory.question_index, []).append(trajectory)
```

分组键不是 `example.id`，也不是 `group_index`：

- `question_index` 表示当前 batch 中同一个题目位置；
- `group_index` 只是该题内部第几个候选；
- 同一道题若因循环取样在 batch 中出现两次，会有两个不同 `question_index`，因而形成两个独立 group。

每组计算：

```text
Aᵢ = Rᵢ - mean(R_group)
```

没有再除以组内标准差。

函数假设输入来自单次 rollout batch。若把多个 batch 的轨迹拼在一起后重新调用，因为每批 `question_index` 都从 0 开始，会把不同批次的题错误合组。

## 6.36 ±1 reward 下 advantage 的通式

假设 group 大小为 `G`，其中 `k` 条正确：

```text
正确 reward = +1
错误 reward = -1
组均值 = (k - (G-k)) / G = (2k-G)/G
```

所以：

```text
正确轨迹 advantage = 1 - mean = 2(G-k)/G
错误轨迹 advantage = -1 - mean = -2k/G
```

例如 `G=8, k=2`：

```text
正确 advantage = 2×6/8 = +1.5
错误 advantage = -2×2/8 = -0.5
```

`k=0` 或 `k=G` 时全组 advantage 为 0；`group_size=1` 也必然为 0。

## 6.37 `progress_callback` 在哪些地方触发

进度单位是“完成的 trajectory”，不是 round 或 tool call。

触发位置：

1. 首轮 `begin_advance()` 直接结束分支时；
2. 首轮沙箱结果因预算失败结束分支时；
3. 后续 `make_request()` 因 prompt 无剩余预算而结束时；
4. 后续 `begin_advance()` 直接结束时；
5. 后续 `finish_advance()` 因 observation 放不下而结束时。

正常继续的工具轨迹不会更新进度，直到之后真正结束。

回调是同步调用并运行在 event-loop 线程中；如果回调阻塞或抛异常，会拖慢或中止 rollout。

## 6.38 一次完整 batch 的数量例子

假设：

```text
questions_per_batch = 2
group_size = 3
```

首轮：

```text
2 个 root
→ 2 个并发 sampling requests
→ 每个 response 3 sequences
→ 6 个 Trajectory branches
```

假设结果：

```text
3 条直接回答并 done
3 条调用工具
```

首轮并发执行 3 个工具调用。后续 round 只为这 3 条活跃轨迹分别创建单样本请求。

若下一轮 2 条回答、1 条继续调用工具，则再下一 round 只剩 1 条活跃轨迹。最终仍返回 6 条轨迹，再按 `question_index` 分成两个各 3 条的 reward group。

## 6.39 rollout 输出如何被 `train.py` 消费

返回值：

```python
list[Trajectory]
```

每条轨迹最关键的训练字段：

```text
turns[*].prompt_tokens
turns[*].completion_tokens
turns[*].logprobs
advantage
```

以两轮轨迹为例：

```text
turn 1: prompt=P₁, completion=C₁, logprobs=L₁
turn 2: prompt=P₁+C₁+D₁, completion=C₂, logprobs=L₂
```

`build_datum()` 会识别：

```text
第 1 轮新增 context = P₁
第 2 轮新增 observation = D₁
```

合成：

```text
full_tokens = P₁ + C₁ + D₁ + C₂

old logprobs = 0...0 + L₁ + 0...0 + L₂
advantages   = 0...0 + A...A + 0...0 + A...A
```

其中 `A=trajectory.advantage`。第七章会继续讲右移和张量字段。

## 6.40 关键不变量清单

```text
1. len(completion_tokens) == len(logprobs)

2. 后续 prompt 必须满足：
   P(i+1) = P(i) + C(i) + D(i)

3. C(i) 始终使用 sampler 返回的真实 token，不能从 text 重编码。

4. AssistantTurn 在协议解析前写入，实际采样动作不会丢失。

5. 首轮每道有效题应产生 group_size 个分支。

6. 首轮请求 num_samples=group_size，后续请求 num_samples=1。

7. prompt、系统补 closing、tool observation、下一轮前缀不是模型动作。

8. 只有 completion token 使用真实旧 logprob 和轨迹 advantage。

9. 一次采样所请求的 prompt+completion 不超过 rollout 总预算，
   前提是 sampler 遵守 max_tokens。

10. 所有轨迹先得到 reward，再按同题组得到 advantage。
```

## 6.41 重点边界与源码疑点

### 边界 1：`fit_tool_content()` 可能在长度 65/66 陷入不再缩短

截断长度近似递推：

```text
new_len = 20 字符提示前缀 + floor(old_len × 0.7)
```

对于某些短长度，例如 65 或 66，新长度可能等于旧长度；而失败退出条件是 `len(content) <= 64`。如果此时 observation 仍放不进总预算，循环可能无法收敛。这是应在本地测试任务中覆盖的真实疑点。

### 边界 2：初始 prompt 已超预算时题目会消失

首轮 root 的 `make_request()` 返回 `None` 后，它不会分叉，也不会加入最终 `trajectories`。于是返回轨迹数可能少于：

```text
len(examples) × group_size
```

且首轮进度不会为该题补上 `group_size`。

### 边界 3：恰好占满预算可能以空 `final_text` 评分

`fit_tool_content()` 允许 `len(next_prompt) == max_trajectory_tokens`。它成功接入 tool message 后，下一轮 `make_request()` 发现剩余为 0，只设置 `done=True`，没有为 `final_text` 赋值；最终空字符串得到 `-1`。

### 边界 4：采样前没有为 tool wrapper 预留空间

assistant completion 可以用完所有剩余预算。若它恰好是工具调用，后面的 closing、tool wrapper 和 observation 必然无法装入，只能事后截断或终止。

### 边界 5：rollout 上限与训练上限可能不一致

```text
RolloutConfig.max_trajectory_tokens → 命令行可调
train.py MAX_TRAIN_CONTEXT_TOKENS   → 固定为 8192
```

若把 rollout 上限调到 8192 以上，轨迹可能生成成功，却在 `build_datum()` 阶段报超长错误。

### 边界 6：`sequence.text` 与 tokens 的一致性依赖服务

源码只检查 token/logprob 等长，不检查 text 是否能从这些 token 严格得到。正常 sampler 应满足契约，但若不一致，协议执行的是一段文本，训练的却可能是另一组 token。

### 边界 7：基础设施异常没有局部降级

`asyncio.gather()` 默认遇到任一采样、线程或进程基础设施异常就向上抛出；没有 retry 或“只让单条轨迹失败”的机制。普通 Python 语法错误和超时已经由 sandbox 包装，不属于此类异常。

### 边界 8：配置没有统一的正值校验

例如 `group_size=1` 可运行但 advantage 恒为 0；`max_assistant_turns<=0` 仍会先采样一次；`max_workers=0` 可能导致工具执行永久等待。正常实验必须使用合理正值配置。

### 边界 9：空输入安全返回空列表

`examples=[]` 时不会创建请求、不会进入 while，最后返回 `[]`。训练入口已在更早处拒绝空训练数据。

## 6.42 本章对象与状态账本

| 阶段 | 主要对象 | 关键变化 |
| --- | --- | --- |
| 创建根轨迹 | `Trajectory root` | 写入 example、question_index、初始 messages |
| 构造请求 | `SampleRequest` | 确定真实 prompt、num_samples、max_tokens、seed |
| 消费采样 | `AssistantTurn` | 保存 P/C/L/text，turns 加一 |
| 判断动作 | `ParsedAssistant` | tool 继续；answer/invalid 或上限则结束 |
| 等待工具 | `PendingExecution` | 保存代码、轨迹引用和 token 续接上下文 |
| 写回工具 | `Trajectory` | 追加 tool message、next prompt、code_calls 加一 |
| 轨迹终止 | `Trajectory` | done=True，保存 final_text |
| 结果评分 | `Trajectory` | reward、correct、valid_format |
| 组内比较 | `Trajectory` | advantage = reward - group mean |

## 6.43 本章小结

用一句话概括 `rollout.py`：

> 它先用共享首轮 prompt 为每道题分叉出一个采样组，再按 round 并发推进各条独立轨迹；每轮保存真实 prompt、completion 和旧 logprob，合法工具调用经沙箱变成 observation 并连续接回 token 历史，直到回答、协议错误或预算耗尽，最后统一计算 outcome reward 和同题组 advantage。

读完本章应能回答：

1. 为什么首轮使用 `num_samples=group_size`，后续却是每轨迹 `num_samples=1`？
2. `question_index`、`group_index` 和 `trajectory_index` 有什么区别？
3. `AssistantTurn` 为什么同时保存 token、logprob 和 text？
4. 为什么 `begin_advance()` 要先写入 turn，再解析输出？
5. 默认 6 个 assistant turns 时，为什么第 6 轮工具调用不会执行？
6. `code_calls` 为什么可能小于实际沙箱执行次数？
7. `fit_tool_content()` 同时检查哪两类预算？
8. “round 内并发、round 间屏障”是什么意思？
9. 两轮工具轨迹为什么满足 `P₂=P₁+C₁+D₁`？
10. rollout 返回后按什么字段分组计算 advantage？
11. 哪些终止条件会设置 `done=True`？
12. 哪些边界可能导致轨迹消失、空 final_text 或截断循环不收敛？
