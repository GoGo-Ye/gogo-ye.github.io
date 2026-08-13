---
title: ReTool 源码精读05
publishDate: 2026-08-12T12:00:00
description: '第五章：`reward.py`——最终答案与稀疏奖励'
tags:
  - 笔记
---
# 第五章：`reward.py`——最终答案与稀疏奖励

源码位置：`https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/05-retool/reward.py`

## 5.1 先抓住本章主线

一条轨迹结束后，`reward.py` 只接收两个字符串：

```python
text = trajectory.final_text
reference = trajectory.example.answer
```

然后执行：

```text
final_text
    ↓ 只保留末尾 300 个字符
回答窗口
    ↓ extract_last_boxed()
最后一个配平的 \boxed{...} 内容
    ↓ answers_equivalent()
与参考答案做数学等价判定
    ↓ score_answer()
RewardResult
    ├── reward: +1.0 / -1.0
    ├── correct: True / False
    ├── valid_format: True / False
    └── answer: 抽取到的原始答案或 None
```

这是典型的 **outcome-only reward**：只看轨迹最后交付的结果，不给中间推理、代码调用或工具纠错过程单独打分。

## 5.2 reward 看的是哪段文本

`rollout.py` 在轨迹结束时保存：

```python
trajectory.final_text = text
```

其中 `text` 是最后一轮 assistant 输出，而不是完整的 `messages`，也不是所有 assistant turn 的拼接。

假设轨迹经历三轮：

```text
第 1 轮 assistant：生成 Python 工具调用
tool observation：返回 5
第 2 轮 assistant：工具调用报错后重试
tool observation：返回 3
第 3 轮 assistant：最终回答 \boxed{3}
```

reward 实际只检查：

```text
第 3 轮 assistant 的 final_text
```

它不会直接检查：

- 前两轮的推理是否合理；
- Python 代码是否优雅或高效；
- 工具调用次数；
- 中间是否出现异常；
- observation 是否被正确解释；
- 最终答案是靠工具算出还是碰巧猜中。

只要最终结果通过验证，就得到正奖励。

## 5.3 `RewardResult`：一次评分的四个结果

```python
@dataclass(frozen=True)
class RewardResult:
    reward: float
    correct: bool
    valid_format: bool
    answer: str | None
```

四个字段不要混为一谈：

| 字段 | 回答的问题 |
| --- | --- |
| `answer` | 从最后一个 `\boxed{...}` 中抽出了什么原文？ |
| `valid_format` | 是否找到了一个花括号完整闭合的 `\boxed{...}`？ |
| `correct` | 抽取答案是否与参考答案数学等价？ |
| `reward` | 最终用于 RL 的标量是 `+1.0` 还是 `-1.0`？ |

对象使用 `frozen=True`，因为一次评分结果生成后不应被修改。

### 四个字段的关系

当前源码只有三种实际组合：

| 情况 | `answer` | `valid_format` | `correct` | `reward` |
| --- | --- | ---: | ---: | ---: |
| 没找到完整 boxed | `None` | `False` | `False` | `-1.0` |
| 找到 boxed，但答案错误 | 抽取字符串 | `True` | `False` | `-1.0` |
| 找到 boxed，且答案正确 | 抽取字符串 | `True` | `True` | `+1.0` |

所以：

```text
correct=True  ⇒ valid_format=True
```

但反过来不成立：格式合法不代表数学正确。

## 5.4 为什么只检查末尾 300 个字符

常量：

```python
ANSWER_WINDOW_CHARS = 300
```

评分入口首先执行：

```python
text[-ANSWER_WINDOW_CHARS:]
```

也就是只保留最终回答最后 300 个 Python 字符。

这样做对齐官方 DAPO 严格 boxed 验证规则，也强化了“最终答案应该出现在回答结尾”的格式约束。模型不能在很早的位置随手写一个 boxed，然后在后面继续输出大量无关内容，仍指望早期答案被评分器采用。

例如：

```text
开头：\boxed{3}
中间和结尾：超过 300 个字符的其他内容
```

早期 boxed 已经落在窗口之外，评分器找不到它，结果为：

```python
RewardResult(-1.0, False, False, None)
```

### 这里是字符窗口，不是 token 窗口

`300` 表示 Python 字符串切片长度，不是 300 个 tokenizer token，也不是 300 bytes。

因此：

```text
reward 的答案窗口 → 300 characters
rollout 的上下文预算 → tokenizer tokens
sandbox 的读取上限 → bytes / characters
```

三者单位不同。

### 窗口边界的一个细节

如果 `\boxed{` 的开头刚好位于最后 300 字符之前，即使大部分答案内容落在窗口内，切片后也找不到完整 marker，仍会判为格式非法。

## 5.5 `extract_last_boxed()`：先找最后一个 marker

函数先定义：

```python
marker = "\\boxed{"
index = text.rfind(marker)
```

注意 Python 字符串里的：

```python
"\\boxed{"
```

实际文本是：

```text
\boxed{
```

因为反斜杠在 Python 字符串字面量中需要转义。

`rfind()` 从右侧寻找最后一次出现的位置：

```text
推理中写了 \boxed{2}，最后修正为 \boxed{3}
                                      ↑
                               选择最后一个 marker
```

如果完全没有找到：

```python
index == -1
```

函数返回 `None`。

## 5.6 为什么不能用简单正则 `\boxed{(.*?)}`

boxed 答案内部可能包含嵌套花括号，例如：

```text
\boxed{\frac{1}{2}}
```

如果只找到第一个 `}`，会错误地抽取：

```text
\frac{1
```

所以源码使用花括号深度计数，而不是简单地寻找第一个右花括号。

初始化：

```python
start = index + len(marker)
depth = 1
```

`depth=1` 表示已经进入 `\boxed{` 的最外层左花括号。

随后逐字符扫描：

```python
if text[position] == "{":
    depth += 1
elif text[position] == "}":
    depth -= 1
    if depth == 0:
        return text[start:position]
```

## 5.7 用嵌套答案走一遍深度计数

输入：

```text
\boxed{\frac{1}{2}}
```

只看花括号变化：

| 字符 | 含义 | `depth` |
| --- | --- | ---: |
| `\boxed{` 的 `{` | 已在扫描前计入 | 1 |
| `\frac{` 的 `{` | 进入分子 | 2 |
| 分子后的 `}` | 离开分子 | 1 |
| 分母前的 `{` | 进入分母 | 2 |
| 分母后的 `}` | 离开分母 | 1 |
| 最后的 `}` | 关闭 boxed | 0 |

当 `depth` 第一次回到 0，函数返回外层花括号之间的内容：

```text
\frac{1}{2}
```

因此它支持任意层数的配平花括号。

## 5.8 “最后一个 boxed”的严格含义

源码先用 `rfind()` 选中最后一个 `\boxed{`，然后只尝试配平这一个 marker。

### 两个都完整

```text
First guess: \boxed{2}
Correction: \boxed{3}
```

抽取结果是：

```text
3
```

### 最后一个不完整

```text
Earlier: \boxed{3}
Later malformed: \boxed{4
```

函数不会退回使用前面的完整 `\boxed{3}`，而是扫描最后一个 marker 后发现无法闭合，最终返回 `None`。

这体现了严格的“最终答案优先”规则：最后一次声明如果格式坏了，不用更早的候选兜底。

## 5.9 哪些格式会被认为有效

评分器寻找的是精确字符串：

```text
\boxed{
```

因此：

```text
\boxed{3}     → 可以识别
\boxed {3}    → 不能识别，命令和左花括号间多了空格
boxed{3}      → 不能识别，缺少反斜杠
\fbox{3}      → 不能识别，命令不同
\boxed{3      → 不能识别，花括号未闭合
```

但以下形式会被认为 `valid_format=True`：

```text
\boxed{}
```

因为花括号完整闭合，抽取结果是空字符串 `""`，不是 `None`。它通常无法通过数学等价验证，所以会得到：

```text
valid_format=True
correct=False
reward=-1.0
```

因此这里的 `valid_format` 更准确地说是：

> 找到了语法上可配平的 boxed 容器。

它不保证容器内容非空、可解析或正确。

## 5.10 `answers_equivalent()`：不是字符串相等

源码没有直接执行：

```python
prediction == reference
```

而是：

```python
verify(
    parse(f"${reference}$"),
    parse(f"${prediction}$"),
)
```

流程是：

```text
参考答案字符串 ── 加 $...$ ── parse() ──┐
                                         ├─ verify() → bool
预测答案字符串 ── 加 $...$ ── parse() ──┘
```

给两边加 `$...$`，是把它们作为 LaTeX 数学表达式交给 `math_verify` 解析。

这样做的目标是识别数学等价，而不仅是字符完全相同。例如概念上：

```text
1/2、0.5、\frac{1}{2}
```

可能被解析为同一个数学值。具体支持范围由 `math_verify` 的解析与验证规则决定，不能把它理解为任意自然语言答案都能可靠比较。

### 两边都会先 `strip()`

在 `score_answer()` 中调用：

```python
answers_equivalent(answer.strip(), reference.strip())
```

所以 boxed 内容和参考答案首尾的普通空白不影响验证。但 `RewardResult.answer` 保存的是抽取到的原始内容，没有执行 `strip()`，便于日志中观察模型真实格式。

## 5.11 为什么捕获所有验证异常

```python
try:
    return bool(verify(...))
except Exception:
    return False
```

模型可能生成怪异或无法解析的数学文本。若其中一个异常直接终止整个 rollout batch，会让一次坏回答拖垮整个训练 step。

当前策略是：

```text
解析或验证异常
    → 视为答案不等价
    → correct=False
    → reward=-1.0
```

这样 reward 函数对任意模型输出都尽量返回确定结果。

代价是异常原因没有被记录；调试某类答案为何全部判错时，需要单独复现 `parse/verify` 过程。

## 5.12 `score_answer()` 的完整决策树

源码可以展开为：

```text
输入 text 和 reference
    │
    ├─ text 只取最后 300 字符
    │
    ├─ 提取最后一个完整 \boxed{...}
    │      │
    │      └─ 没找到
    │           → answer=None
    │           → valid_format=False
    │           → correct=False
    │           → reward=-1.0
    │
    └─ 找到 answer
           │
           ├─ math_verify 判定等价
           │      → valid_format=True
           │      → correct=True
           │      → reward=+1.0
           │
           └─ 不等价或验证异常
                  → valid_format=True
                  → correct=False
                  → reward=-1.0
```

对应代码：

```python
answer = extract_last_boxed(text[-ANSWER_WINDOW_CHARS:])
if answer is None:
    return RewardResult(-1.0, False, False, None)

correct = answers_equivalent(answer.strip(), reference.strip())
return RewardResult(
    1.0 if correct else -1.0,
    correct,
    True,
    answer,
)
```

## 5.13 四个完整评分例子

假设参考答案：

```python
reference = "3"
```

### 例 1：格式和答案都正确

```text
Solving the equation gives x=3.
\boxed{3}
```

结果：

```python
RewardResult(
    reward=1.0,
    correct=True,
    valid_format=True,
    answer="3",
)
```

### 例 2：格式正确但答案错误

```text
\boxed{4}
```

结果：

```python
RewardResult(-1.0, False, True, "4")
```

### 例 3：数学答案正确但格式错误

```text
The answer is 3.
```

结果：

```python
RewardResult(-1.0, False, False, None)
```

这是 `prepare_data.py` 必须删除旧 `Answer: 3` 模板的原因：reward 只接受 boxed 容器。

### 例 4：先答错，最后修正

```text
Initially I obtained \boxed{4}, but checking again gives \boxed{3}.
```

最后一个 boxed 是 `3`，因此得到 `+1.0`。

## 5.14 `valid_format` 不等于完整协议合法

这是源码中容易忽略的边界。

`valid_format` 只来自：

```python
extract_last_boxed(...) is not None
```

它没有调用 `parse_assistant()`，也不知道 assistant 输出之前是否被归类为 `answer`、`tool` 或 `invalid`。

例如一个非法输出：

```text
<tool_call>格式损坏...</tool_call>
\boxed{3}
```

`parse_assistant()` 可能把它归为 `invalid` 并结束轨迹；但 `score_answer()` 仍可能找到末尾的 `\boxed{3}`，并在数学等价时给出 `+1.0`。

因为 `rollout.py` 当前没有把“协议 invalid”单独传给 reward，也没有强制覆盖为负奖励。

所以当前实现中的两个概念必须分开：

```text
工具协议合法性 → parse_assistant() 决定状态转移
boxed 格式合法性 → reward.py 的 valid_format
```

通常模型不会频繁产生“非法工具协议 + 正确 boxed”的混合输出，但从严格源码语义看，`invalid` 并不自动保证 reward 为 `-1`。这是后续测试任务值得覆盖的边界用例。

## 5.15 reward 如何写回 `Trajectory`

所有轨迹结束后，`rollout.py` 调用：

```python
def score_trajectory(trajectory):
    result = score_answer(
        trajectory.final_text,
        trajectory.example.answer,
    )
    trajectory.reward = result.reward
    trajectory.valid_format = result.valid_format
    trajectory.correct = result.correct
```

数据对应关系：

```text
RewardResult.reward       → Trajectory.reward
RewardResult.valid_format → Trajectory.valid_format
RewardResult.correct      → Trajectory.correct
```

`RewardResult.answer` 没有写进 `Trajectory`。抽取答案主要用于调用方即时调试；当前轨迹对象只长期保留 reward、格式标志和正确性标志。

评分发生在全部轨迹完成之后：

```python
for trajectory in trajectories:
    score_trajectory(trajectory)
assign_group_advantages(trajectories)
```

顺序必须是先得到每条轨迹 reward，再按同题组计算 advantage。

## 5.16 reward 与 advantage 不是同一个量

`reward.py` 只产生：

```text
+1.0 或 -1.0
```

`rollout.py` 随后按同一道题的采样组计算：

```python
advantage_i = reward_i - mean_reward_of_group
```

假设同一道题采样 4 条轨迹：

```text
reward = [+1, -1, -1, -1]
```

组均值：

```text
mean = (-2) / 4 = -0.5
```

所以：

```text
正确轨迹 advantage = +1 - (-0.5) = +1.5
错误轨迹 advantage = -1 - (-0.5) = -0.5
```

可以看到 advantage 可能超出 reward 的 `[-1, +1]` 范围。

### 一半正确、一半错误

```text
reward    = [+1, +1, -1, -1]
mean      = 0
advantage = [+1, +1, -1, -1]
```

### 全部正确

```text
reward    = [+1, +1, +1, +1]
mean      = +1
advantage = [0, 0, 0, 0]
```

### 全部错误

```text
reward    = [-1, -1, -1, -1]
mean      = -1
advantage = [0, 0, 0, 0]
```

这就是 degenerate group：同组没有相对差异，无法告诉策略“哪条轨迹比同题的其他轨迹更好”。即使全组 reward 都是 `+1`，advantage 仍全为 0，该组也不产生当前实现中的 PPO 更新信号。

## 5.17 稀疏奖励“稀疏”在哪里

一条多轮轨迹可能生成上千个 assistant token，并进行多次工具交互，但 reward 只在轨迹结束时计算一次：

```text
turn 1 assistant code       → 没有即时 reward
turn 1 tool observation     → 没有即时 reward
turn 2 assistant correction → 没有即时 reward
turn 2 tool observation     → 没有即时 reward
turn 3 final answer         → 整条轨迹得到 ±1
```

后续构造训练 Datum 时，同一条轨迹的所有 assistant completion token 会共享轨迹 advantage，而所有 prompt/tool observation token 的 advantage 为 0。

因此 reward 不会直接指出：

- 哪一行代码有帮助；
- 哪一步推理错误；
- 哪次工具调用应该删除；
- 哪个 assistant token 最关键。

模型只能通过大量同题采样和组内相对比较，逐渐提高产生成功完整轨迹的概率。

## 5.18 格式奖励和正确性奖励没有分开加权

当前实现不是：

```text
有 boxed → +0.2
答案正确 → 再 +0.8
```

而是严格二值规则：

```text
boxed 合法且数学正确 → +1
其他所有情况         → -1
```

所以格式合法但答案错误，与完全没有 boxed，在 reward 数值上都是 `-1`。

`valid_format` 单独记录只是为了日志分析：可以区分模型到底是“不会遵守输出格式”，还是“格式已经学会但数学仍算错”。

例如：

```text
format rate 高、correct rate 低
    → 模型会输出 boxed，但解题能力不足

format rate 低
    → 协议遵循本身仍有问题
```

## 5.19 `eval.py` 为什么还有一个 boxed 提取函数

`eval.py` 为 AIME 评测另外定义了 `extract_last_boxed()`，主要用于生成可读的 `predicted_answer` 字段和 text-only 对照评测。

两个实现并不完全相同：

### `reward.py`

```text
只找精确的 \boxed{
只检查 final_text 最后 300 字符
若最后一个 marker 不闭合，不回退更早 marker
用 math_verify 做通用数学等价验证
```

### `eval.py`

```text
寻找 \boxed 后再寻找左花括号
不限制最后 300 字符
最后一个候选不闭合时会向前尝试更早候选
AIME 答案再规范化为整数进行比较/展示
```

在 retool 评测模式下，真正的 `trajectory.correct` 和 `trajectory.valid_format` 仍来自 `reward.py`；`eval.py` 自己抽取的答案主要用于结果文件中的展示字段。

因此某些极端格式下，JSONL 中展示的 `predicted_answer` 与轨迹的 `formatted/correct` 标志理论上可能出现差异。正式比较指标时要看 summary 使用了哪个字段。

## 5.20 重点与疑难点

### 重点 1：reward 只看最终 outcome

中间代码是否报错不直接受罚；只要模型利用错误反馈完成自我纠正，最终仍可获得 `+1`。

### 重点 2：格式合法与答案正确是两个维度

`valid_format=True` 只说明存在可配平的 boxed 容器；`correct=True` 才表示数学验证通过。

### 重点 3：reward 与 advantage 分两步产生

单条轨迹先得到 `±1` reward；同题多轨迹再中心化得到 advantage。训练真正使用的是 advantage。

### 疑难点 1：最后一个 malformed boxed 会覆盖前面的正确 boxed

评分器不会向前回退，这使最后答案格式具有严格优先级。

### 疑难点 2：数学解析异常被静默当成错误

这能保护训练循环，但会隐藏具体解析失败原因。调试时应保存或复现被抽取的 `answer`。

### 疑难点 3：协议 invalid 不会自动覆盖 reward

当前 reward 函数只验证 boxed 与数学等价，没有接收工具协议合法性标志。

### 疑难点 4：空 boxed 也算 `valid_format=True`

因为抽取结果是空字符串而不是 `None`，但其正确性通常为 False。

### 疑难点 5：全对组也没有相对训练信号

GRPO 风格的组内中心化关心相对差异；全对与全错都会得到全零 advantage。

## 5.21 本章对象账本

| 阶段 | 对象 | 类型 | 取值或来源 | 是否直接用于 PPO |
| --- | --- | --- | --- | --- |
| 最终输出 | `final_text` | `str` | 最后一轮 assistant 文本 | 间接 |
| 参考答案 | `reference` | `str` | `MathExample.answer` | 不进入模型 |
| 回答窗口 | `text[-300:]` | `str` | final_text 末尾 | 否 |
| 抽取答案 | `answer` | `str \| None` | 最后一个 boxed 内容 | 否 |
| 格式标志 | `valid_format` | `bool` | boxed 是否配平 | 只用于记录 |
| 正确标志 | `correct` | `bool` | math_verify 结果 | 用于指标 |
| 轨迹奖励 | `reward` | `float` | `+1.0 / -1.0` | 用来计算 advantage |
| 组内优势 | `advantage` | `float` | reward 减组均值 | 是，复制到 assistant token |

## 5.22 本章小结

用一句话概括 `reward.py`：

> 它只检查最后一轮 assistant 输出末尾 300 个字符中的最后一个完整 `\boxed{...}`，用 `math_verify` 判断其与参考答案是否数学等价，正确给 `+1`，其余给 `-1`；之后 rollout 再把同题组 reward 中心化为真正用于 PPO 的 advantage。

读完本章应能回答：

1. reward 检查完整多轮历史，还是只检查 `final_text`？
2. 为什么答案窗口使用 300 个字符而不是 token？
3. `extract_last_boxed()` 如何处理 `\frac{1}{2}` 的嵌套花括号？
4. 为什么最后一个 boxed 不完整时不会使用前一个完整 boxed？
5. `valid_format`、`correct`、`reward` 和 `advantage` 有什么区别？
6. 为什么 `\boxed{}` 可能 `valid_format=True` 但 reward 为 `-1`？
7. 为什么答案比较不能只用字符串相等？
8. 一条正确、三条错误的四轨迹组，各自 advantage 是多少？
9. 为什么全组都正确仍可能不产生 PPO 更新信号？
10. 为什么协议 `invalid` 的轨迹在极端情况下仍可能得到正 reward？
