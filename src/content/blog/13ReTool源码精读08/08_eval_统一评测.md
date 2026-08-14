---
title: ReTool 源码精读08
publishDate: 2026-08-14
description: '第八章：`eval.py`——Base/checkpoint 统一评测'
tags:
  - 笔记
---


# 第八章：`eval.py`——Base/checkpoint 统一评测

源码位置：`https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/05-retool/eval.py`

## 8.1 先抓住整章主线

`eval.py` 用同一个评测外壳支持两类模型来源和两种推理模式：

```text
AIME25 Dataset
    ↓
Base model 或 sampler checkpoint
    ↓ 创建统一 SamplingClient
retool 或 text-only
    ↓
每题 N 条 generation
    ↓
problem records
    ↓
Average@N / Pass@N / Format / tool statistics
    ↓
JSONL 最后一行 summary
```

本章最重要的三个结论：

1. Base 与 checkpoint 没有两套评测代码，只在 `model_path` 上不同。
2. ReTool 评测直接复用训练时的多轮 rollout、协议和 reward。
3. `retool` 与 `text-only` 的 prompt、预算和判分并不完全相同，后者不是严格的“只关闭工具”消融。

## 8.2 两个彼此独立的选择轴

不要把“模型来源”和“推理模式”混成一件事：

| 维度 | 选择 1 | 选择 2 |
| --- | --- | --- |
| 模型来源 | `model_path=None`：Base | `model_path=trio://...`：LoRA sampler weights |
| 推理模式 | `mode=retool`：多轮工具轨迹 | `mode=text-only`：单轮无工具 |

因此理论上有四种组合。论文式 Base/checkpoint 横向比较通常固定：

```text
mode = retool
temperature = 1.0
top_p = 0.7
val_n = 12
全部轨迹与沙箱预算相同
```

然后只改变 `model_path`。

“统一评测”指共用同一入口和后续流程，并不是一次命令同时评两个模型。

## 8.3 异步入口与 5 小时采样超时

文件最后执行：

```python
asyncio.run(evaluate(parse_args()))
```

这里的两个名字容易混淆：

```text
asyncio → Python 的异步事件循环
trio    → import pytrio as trio，是远端模型 SDK，不是 Trio 异步库
```

模块导入时还执行：

```python
trio.configure(sampling_timeout=18000)
```

`18000` 秒等于 5 小时。它放宽的是 PyTRIO 单次采样请求超时，不是整场评测的总时限；本地 Python 沙箱仍由独立的 `sandbox_timeout=30` 秒控制。

## 8.4 AIME25 数据如何进入评测

默认路径：

```text
04-opsd/datasets/aime_2025
```

`load_aime25()` 依次检查：

```text
路径存在
→ load_from_disk()
→ DatasetDict 时取 train split
→ 结果必须是 Dataset
→ 必须有 problem / answer 两列
→ 截取前必须恰好有 30 行
→ limit>0 时选择前 min(limit, 30) 题
```

所以这个函数不是通用数学评测加载器。即使只想 `--limit 5`，磁盘上的原始数据仍必须先是完整 30 题。

`limit` 取数据集开头，不随机抽样；小规模 smoke test 的结果可能受题目顺序影响。

## 8.5 Base 与 checkpoint 如何归一成同一种对象

无论评什么模型，核心代码都相同：

```python
sampling_client = await service_client.create_sampling_client_async(
    base_model=args.base_model,
    model_path=args.model_path,
)
tokenizer = sampling_client.get_tokenizer()
```

### Base model

```text
--base-model Qwen/Qwen3.5-4B
不传 --model-path
```

PyTRIO 为基础模型创建 sampling run。

### checkpoint

```text
--base-model Qwen/Qwen3.5-4B
--model-path trio://.../sampler_weights/...
```

这里必须使用第七章中：

```python
save_weights_for_sampler()
```

产生的 sampler weights，而不是 `save_state()` 的训练断点路径。评测只创建 sampling client，不恢复 optimizer，也不创建 training client。

脚本不会从 checkpoint 自动推断基础模型，也不在本地验证 URI 与 base model 是否兼容；这些检查交给 PyTRIO 服务。

## 8.6 `text-only` prompt 是什么

单轮模式构造一条 user message：

```text
<problem>

Please reason step by step, and put your final answer within \boxed{}.
```

然后：

```python
tokenizer.apply_chat_template(
    [{"role": "user", "content": content}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
```

它没有 ReTool 的 system prompt，也没有 `code_interpreter` tool schema。模板先渲染成文本，再由 `tokenizer.encode(..., add_special_tokens=False)` 编码；空 prompt 会直接报错。

## 8.7 `text-only` 如何并发采样

每道题创建一个 coroutine，并通过：

```python
asyncio.gather(...)
```

一起调度。真正的远端采样被 semaphore 包住：

```python
async with semaphore:
    response = await sampling_client.sample_async(
        prompt=...,
        num_samples=args.val_n,
        ...,
    )
```

因此 `--concurrency=15` 限制的是同时在途的“题目请求数”，不是 generation 数。一个请求内部会一次返回该题的 `val_n` 条 completion。

采样参数：

```text
max_tokens = args.max_tokens
seed        = args.seed + 全局题号
temperature = args.temperature
top_p       = args.top_p
top_k       = args.top_k
stop        = eos_token 与 <|im_end|> 去重后的列表
```

`max_tokens=8192` 只限制单次 completion，不包含 prompt。

`gather()` 的完成顺序可以不同，但返回顺序与输入 coroutine 一致，所以最终 problem records 仍按数据集顺序排列。

## 8.8 text-only 的答案抽取与规范化

`extract_last_boxed()` 从整段 completion 向前寻找最后一个花括号配平的 `\boxed...{...}`，支持内部嵌套花括号；若最后一次出现没有闭合，它会继续尝试更早的 boxed。

抽出的文本再经过：

```text
strip
→ 删除逗号与 $
→ 展开一层 \text{...} / \mathrm{...}
→ 必须完全匹配带可选正负号的整数
→ int() 去掉前导零和显式正号
```

例如：

```text
"$\mathrm{007}$" → "7"
"21+21"           → None
"42.0"            → None
```

text-only 的正确性就是：

```python
predicted_answer == ground_truth
```

找到 boxed 就令 `formatted=True`，即使其中不是可规范化的整数。

## 8.9 ReTool 模式如何复用训练状态机

每个 chunk 的数据先适配成：

```python
MathExample(
    id=...,
    question=row["problem"],
    answer=row["answer"],
    data_source="aime_2025",
)
```

再构造与训练同类型的 `RolloutConfig`：

```text
group_size              = val_n
max_code_calls          = 4
max_assistant_turns     = 6
max_trajectory_tokens   = 8192
max_assistant_tokens    = 1024
max_tool_response_tokens= 512
temperature             = 1.0
top_p                   = 0.7
seed                    = 42
```

随后直接调用：

```python
await rollout_batch_async(...)
```

这意味着 ReTool 评测与训练共同使用：

```text
同一个 system prompt 与 tool schema
同一个 token-in/token-out 多轮续接
同一个工具协议解析器
同一个 Python sandbox
同一个轨迹与工具预算逻辑
同一个 reward.py 判分器
```

rollout 仍会计算 reward 和组内 advantage。评测只消费 `correct`、`valid_format` 等字段，advantage 本身被忽略；这是复用训练状态机带来的一点额外工作。

## 8.10 chunk 与状态机内部的并发

默认 30 题按 `chunk_size=5` 切成 6 个 chunk：

```text
chunk 1 完整 rollout → 落盘
chunk 2 完整 rollout → 追加落盘
...
```

chunk 之间严格串行，但一个 chunk 内部有两层并发：

```text
首轮：每题 1 个 sampling request，每个 request 返回 val_n 个分支
后续：每条存活轨迹 1 个单样本 request，同一 round 并发
工具：同一 round 的代码调用并发，再由 sandbox_workers 限流
```

默认 `chunk_size=5, val_n=12` 时：

```text
首轮最多 5 个并发 sampling RPC，每个返回 12 条
后续一轮最多约 60 个单样本 sampling RPC
Python 子进程并发上限为 8
```

rollout 仍是“round 内并发、round 间有屏障”：必须等本轮所有采样和工具结果处理完，才开始下一轮。

## 8.11 `chunk-size` 不是纯性能参数

`evaluate_chunk_retool()` 每处理一个新 chunk，`question_index` 都从 0 重新开始。rollout 的首轮 seed 又是：

```text
base seed + chunk 内 question_index
```

后续 seed 也依赖 chunk 内重新编号的 trajectory index。因此：

```text
不同 chunk 会重复使用相同 seed 序列
改变 chunk-size 会改变同一道全局题目对应的 seed
```

所以比较 Base 与 checkpoint 时必须固定 `chunk-size`。它不仅影响吞吐和活跃轨迹数量，也影响采样结果。

text-only 不存在这个问题：它使用 `seed + 全局题号`，改变 `concurrency` 不改变题目 seed 映射。

## 8.12 两种 mode 不是严格单变量对照

| 维度 | `text-only` | `retool` |
| --- | --- | --- |
| prompt | user 题目 + step-by-step/boxed 指令 | ReTool system prompt + user 题目 + tool schema |
| 回合 | 固定单轮 | 最多 6 个 assistant turns |
| 工具 | 无 | 最多 4 次成功接入的调用 |
| 8192 含义 | 单次 completion 上限 | 整条 prompt/action/observation 轨迹上限 |
| 单轮上限 | 8192 | 默认 1024 |
| stop | EOS + 显式 `<|im_end|>` | protocol 返回的 EOS |
| `top_k` | 生效 | 没有传入 rollout，实际不生效 |
| 正确性 | 整数规范化后字符串相等 | `math_verify` 数学等价 |
| boxed 窗口 | 整段 completion | 最终文本末尾 300 字符 |

因此两者适合作行为对照，却不能把差异全部归因于“是否允许调用工具”。

另一个明确问题是：summary 无条件记录 `top_k`，但 ReTool 的 `RolloutConfig` 没有该字段，`rollout.py` 也没有传给 `SamplingParams`。ReTool 结果文件中的 `top_k` 只是命令行值，不是实际生效配置。

## 8.13 ReTool 的结果如何重新按题分组

`rollout_batch_async()` 返回 chunk 内扁平的：

```python
list[Trajectory]
```

评测脚本用局部 `question_index` 取回每题 group：

```python
group = [
    trajectory
    for trajectory in trajectories
    if trajectory.question_index == question_index
]
```

正常情况下每组恰好有 `val_n` 条。但此路径没有像 text-only 那样显式检查最终长度；若初始 prompt 已经超出轨迹预算，rollout 可能根本没有为该题创建分支，输出仍写请求值 `val_n`，实际 `generations` 却少于 N。

## 8.14 每条 generation 实际保存什么

两种 mode 使用相同字段名，但来源不同：

| 字段 | text-only | retool |
| --- | --- | --- |
| `predicted_answer` | 全文 boxed 的整数规范化结果 | 对 `final_text` 再做同样展示型抽取 |
| `boxed_answer` | 全文最后一个完整 boxed 原文 | 对 `final_text` 的展示型抽取 |
| `correct` | 与整数 ground truth 字符串相等 | `trajectory.correct`，来自 `reward.py` |
| `formatted` | 全文找到 boxed | `trajectory.valid_format`，来自 `reward.py` |
| `code_calls` | 恒为 0 | 成功把 tool observation 接回历史的次数 |
| `turns` | 恒为 1 | assistant 采样轮数 |
| `completion_tokens` | 单轮 completion 长度 | 所有 assistant turn completion 长度之和 |
| `text` | 整段单轮 completion | 只有最后的 `final_text` |

ReTool 的 JSON 不保存完整多轮 transcript、tool observation、每轮 logprob、reward、advantage、停止原因或沙箱错误分类。它适合汇总比较，但不足以还原工具轨迹做逐步调试。

`code_calls` 也不等于 Python 成功次数：代码运行报错或超时，只要反馈成功接回上下文仍会计数；反过来，代码已经执行但 observation 因 token 预算放不下，则不会递增。

## 8.15 ReTool record 中可能出现“字段看似矛盾”

ReTool 的 `correct/formatted` 来自训练 reward：

```text
只看 final_text 最后 300 个字符
要求精确出现 \boxed{
使用 math_verify 判数学等价
```

但 `predicted_answer/boxed_answer` 是 eval.py 为展示重新计算的：

```text
扫描完整 final_text
对 \boxed 与左花括号之间更宽松
只能把答案规范成整数
```

因此可能出现：

```text
boxed_answer 非空，但 formatted = false
→ boxed 位于末尾 300 字符之外，或格式只被宽松抽取器接受

predicted_answer = null，但 correct = true
→ 例如 boxed 中是与整数答案等价的数学表达式
```

对常见的末尾 `\boxed{42}`，两套字段会一致；阅读边界样本时必须以 `correct/formatted` 作为 ReTool 的正式评测口径。

## 8.16 一道题的 JSON record

每题输出一个对象：

```text
type            = "problem"
problem_id      = int(row["id"])；缺失时使用全局题号
problem         = 去首尾空白后的题目
ground_truth    = 规范化整数答案
val_n           = 请求的 N
num_correct     = 本题正确 generation 数
pass_at_n       = 本题是否至少答对一次
generations     = 该题的 generation 明细
```

`problem_id` 强制转 `int`，非数字 ID 会报错；源码不检查 ID 是否重复。

text-only 会拒绝无法规范化为整数的 ground truth。ReTool 仍用原始答案交给 `math_verify`，规范化后的 `ground_truth` 主要用于输出展示；AIME25 的固定整数答案下通常没有差异。

## 8.17 `summarize()` 的五个核心指标

记：

```text
P = problem records 数
G = 实际 generations 总数
C = correct generations 数
H = 至少答对一次的问题数
F = formatted generations 数
```

则：

```text
Average@N = C / G
Pass@N    = H / P
Format    = F / G

MeanCodeCalls = Σ code_calls / G
MeanTurns     = Σ turns / G
```

空分母时统一返回 `0.0`，避免 NaN。正常 30 题、每题 12 条时：

```text
P = 30
G = 360
```

### Average@N 不是多数投票

它只是 360 条 generation 中答对的比例，用 N 次 Monte Carlo 采样估计单次随机生成的正确率，也就是文中所说的 pass@1 估计。

源码没有统计 majority vote，也没有从 N 个答案选一个最终答案。

### Pass@N 是“至少一次成功”

某题 12 条里只要有 1 条正确，该题的 `pass_at_n=True`。因此它衡量的是采样 N 次能否覆盖到正确解，而不是单次输出的稳定性。

在 30 题上：

```text
Average@12 每多 1 条正确，变化约 1/360 = 0.278 个百分点
Pass@12 每多通过 1 题，变化     1/30  = 3.33 个百分点
```

这解释了为什么 Pass@12 曲线比 Average@12 更容易出现明显跳动。

若异常情况下各题实际 generation 数不同，`Average@N` 会按 generation 加权，不再等于各题正确率的等权平均；名称中的 N 也只是请求值。

## 8.18 summary 保存了什么，又漏了什么

最后一行 `type="summary"` 保存：

```text
dataset / mode
base_model / model_path
temperature / top_p / top_k
val_n
problems / generations
average_at_n / pass_at_n / format_rate
mean_code_calls / mean_turns
correct_generations / passed_problems
```

其中 `dataset` 固定写成：

```text
yentinglin/aime_2025
```

它没有记录：

```text
seed
dataset_path / limit
chunk_size / concurrency
max_tokens
全部 ReTool turn、tool、trajectory token 预算
sandbox timeout / workers
sampling timeout
```

所以仅靠 summary 不能完整复现实验。特别是 `chunk_size` 会改变 seed，而它恰好没有被记录；`top_k` 被记录，却在 ReTool 模式中没有生效。

## 8.19 JSONL 为什么采用“每题一行 + summary”

成功文件结构：

```text
line 1   → problem 0
line 2   → problem 1
...
line P   → problem P-1
line P+1 → summary
```

`ensure_ascii=False` 让题目与模型文本直接以 Unicode 写入。第九章的 `analysis.py` 会寻找 `type="summary"` 的记录来画 checkpoint 对比图。

两种 mode 的落盘时机不同。

### text-only

```text
所有题 gather 成功
→ 一次性用 w 模式写全部 problem records
→ 追加 summary
```

任一题异常都会让 gather 抛出，当前运行没有部分结果落盘。

### ReTool

```text
首 chunk 完成 → w 模式覆盖并写入
后续 chunk    → a 模式追加
全部完成      → 追加 summary
```

长评测中途失败时，已完成 chunk 得以保留，但文件没有 summary，不能被后续分析当成完整评测。源码也没有断点续评；重新运行时首个成功 chunk 会覆盖旧文件。

如果在首个 chunk 完成前就失败，写文件尚未发生，旧的同名结果甚至可能原样保留。为 Base 与各 checkpoint 使用不同 `--output` 很重要。

## 8.20 进度条和终端日志的口径

text-only 进度单位是 problem：每道题的 N 条结果全部返回后加一。

ReTool 进度单位是 trajectory，总量为：

```text
len(dataset) × val_n
```

若某题因初始 prompt 超预算没有生成完整 group，进度条可能无法走满。

ReTool 每个 chunk 还打印：

```text
当前 chunk 正确 generation 数
sandbox.stats.metrics()
```

sandbox 在所有 chunk 间复用，其成功率、错误率、超时率和延迟都是截至当前的累计值，不是单 chunk 指标。这些沙箱指标只打印到终端，不写入 JSONL。

本脚本没有 SwanLab 初始化或日志上报；持久化结果只有 JSONL。

## 8.21 参数按 mode 分组

| 参数组 | 关键参数 | 实际作用域 |
| --- | --- | --- |
| 通用 | base model、model path、val-n、temperature、top-p、seed、output | 两种 mode |
| text-only | concurrency、max-tokens、top-k | 仅 text-only |
| ReTool | chunk-size、code/turn/token 预算、sandbox 参数 | 仅 ReTool |

解析器只显式检查：

```text
val_n / concurrency / max_tokens / chunk_size >= 1
limit >= 0
temperature >= 0
```

它会检查当前 mode 根本不用的参数，例如 ReTool 模式下 `concurrency=0` 仍会报错；反过来，`top_p`、ReTool 各预算和 `sandbox_workers` 没有完整范围校验。

源码帮助中的合法模式是：

```text
retool
text-only
```

项目 `readme.md` 有一处写成 `--mode text`，与 argparse choices 不一致，应以代码中的 `text-only` 为准。

## 8.22 关键不变量

```text
1. Base 与 checkpoint 最终都必须变成 SamplingClient。

2. tokenizer 必须从该 SamplingClient 获取。

3. text-only 的每个采样响应必须恰好返回 val_n 条 sequence。

4. 正常 ReTool 每题也应有 val_n 条 Trajectory，但源码未在汇总处断言。

5. Base/checkpoint 公平比较必须固定 mode、采样参数、预算与 chunk-size。

6. ReTool 的正式 correct/format 必须以 reward.py 写入的轨迹字段为准。

7. 完整 JSONL 必须包含 type=summary 记录。
```

## 8.23 关键边界与源码疑点

### 边界 1：跨 mode 的分数不完全同标尺

text-only 使用全文整数匹配，ReTool 使用末尾 300 字符的数学等价判定。直接比较两者时，差异包含 prompt、预算和 scorer，而不只有工具能力。

### 边界 2：ReTool 的 `top_k` 静默无效

summary 会记录它，但 rollout 没有接收它。读取结果时不能据此认为 ReTool 使用了 top-k sampling。

### 边界 3：chunk-size 改变随机性

局部 question/trajectory index 每个 chunk 重置，导致 seed 复用和映射变化。它必须纳入实验配置控制。

### 边界 4：没有 N 条结果也可能继续汇总

ReTool 路径没有最终 cardinality check；summary 以实际 generations 为分母，可能掩盖缺样。

### 边界 5：基础设施异常没有 retry

任一 sampling client 创建或采样异常都会向上抛出。ReTool 只保留此前完整 chunk；text-only 当前运行不保留部分结果。

### 边界 6：结果元数据不足

summary 没有保存 seed、数据范围和预算，默认 output 又不区分 Base/checkpoint。可靠实验需要命令、日志或文件名额外承担配置记录。

## 8.24 本章对象账本

| 阶段 | 输入 | 输出 |
| --- | --- | --- |
| 数据加载 | AIME25 磁盘目录 | 30 题或前 N 题 Dataset |
| 模型创建 | base model + 可选 model path | SamplingClient |
| text-only | Dataset row | N 条单轮 sequence |
| ReTool 适配 | Dataset row | MathExample |
| ReTool rollout | chunk + RolloutConfig | 多轮 Trajectory 列表 |
| 逐题汇总 | sequence/trajectory group | problem record |
| 全局汇总 | problem records | summary record |
| 持久化 | problem + summary | JSONL |

## 8.25 本章小结

用一句话概括 `eval.py`：

> 它把 Base 和 LoRA sampler weights 统一成同一种 SamplingClient，再按 text-only 单轮路径或训练同构的 ReTool 多轮状态机为 AIME25 每题采样 N 次，写出逐 generation 记录并聚合 Average@N、Pass@N、格式率和工具统计；公平比较依赖调用者固定全部有效配置，并正确处理两种 mode 的 prompt、预算与 scorer 差异。

读完本章应能回答：

1. Base 与 checkpoint 在源码中唯一的分叉是什么？
2. 为什么 checkpoint 应传 sampler weights 而不是 state？
3. text-only 的 `concurrency` 限制请求数还是 generation 数？
4. ReTool 的 chunk 内有哪些并发层次？
5. 为什么改变 `chunk-size` 会改变结果？
6. text-only 与 ReTool 的 8192 token 各限制什么？
7. 为什么 ReTool 的 `predicted_answer` 可能为 null，但 `correct=true`？
8. Average@N 与 Pass@N 的分母分别是什么？
9. Average@12 是否使用多数投票？
10. 中途失败后两种 mode 会留下什么文件？
11. 哪个采样参数在 ReTool summary 中被记录却没有生效？
12. 为什么只拿 summary 仍不足以完整复现实验？
