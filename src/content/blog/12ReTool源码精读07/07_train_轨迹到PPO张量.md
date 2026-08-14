---
title: ReTool 源码精读07
publishDate: 2026-08-13T12:00:00
description: '第七章：`train.py`——轨迹到 PPO 张量'
tags:
  - 笔记
---

# 第七章：`train.py`——轨迹到 PPO 张量

源码位置：`https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/05-retool/train.py`

## 7.1 先抓住整章主线

第六章返回的是逐轮轨迹对象，第七章要完成两次转换：

```text
Trajectory.turns
  └── 多轮 prompt / completion / old logprob
          ↓ build_datum()
一条连续、右移对齐的 PPO Datum
          ↓ pack_micro_batches()
若干满足显存代理预算的 micro-batch
          ↓ forward_backward() × K
累计整个 rollout batch 的梯度
          ↓ optim_step() × 1
更新一次 LoRA 参数
```

最关键的三句话：

1. 一条多轮轨迹只构造一个 `Datum`，不是每个 assistant turn 一个。
2. prompt 和 tool observation 保留在模型上下文中，但 advantage 为零，不参与策略梯度。
3. 一个训练 step 可以有多次 `forward_backward()`，却最多只有一次 `optim_step()`。

## 7.2 四个训练常量

```python
MAX_TRAIN_CONTEXT_TOKENS = 8192
MAX_MICRO_BATCH_ITEMS = 32
MAX_MICRO_BATCH_PADDED_TOKENS = 64_000

PPO_LOSS_CONFIG = {
    "clip_low_threshold": 0.8,
    "clip_high_threshold": 1.28,
}
```

| 常量 | 限制对象 |
| --- | --- |
| `MAX_TRAIN_CONTEXT_TOKENS` | 单条右移后训练输入最多 8192 token |
| `MAX_MICRO_BATCH_ITEMS` | 一次手工 micro-batch 最多 32 条 Datum |
| `MAX_MICRO_BATCH_PADDED_TOKENS` | `样本数 × 批内最大长度` 最多 64,000 |
| PPO clip | 新旧策略概率比裁剪到 `[0.8, 1.28]` |

前两个 token 上限不要混淆：rollout 的 `max_trajectory_tokens` 可以由命令行修改，训练 Datum 的 8192 是源码固定值。

## 7.3 `TrainingDatum` 为什么还要包一层

```python
class TrainingDatum:
    def __init__(self, datum: trio.Datum, num_tokens: int) -> None:
        self.datum = datum
        self.num_tokens = num_tokens
```

它只是把两个东西放在一起：

```text
datum      → 真正提交给 PyTRIO 的模型输入与 PPO 字段
num_tokens → 右移后真实序列长度，供排序、装箱和指标统计使用
```

`num_tokens` 包含被 mask 的 prompt 和 observation，所以它不等于真正产生 loss 的 token 数。

## 7.4 `build_datum()` 的数学目标

对第 `i` 个 assistant turn，记：

```text
Pᵢ = turn.prompt_tokens
Cᵢ = turn.completion_tokens
Lᵢ = turn.logprobs
A  = trajectory.advantage
```

两轮轨迹在 rollout 中满足前缀扩展关系：

```text
P₂ = P₁ + C₁ + D₂
```

其中 `D₂` 是两次模型动作之间新增的非模型 token。它通常包含：

```text
模板补齐的 assistant closing
+ tool 消息边界
+ stdout / stderr
+ 下一轮 assistant generation prefix
```

所以源码变量 `delta_observation` 的名字略窄：第一轮它其实是整个初始 prompt，后续它也不只有 observation 正文。

`build_datum()` 最终拼成：

```text
F = P₁ + C₁ + D₂ + C₂ + D₃ + C₃ + ...
```

也就是保留实际生成事实的一条连续自回归序列。

## 7.5 如何找出每轮新增的环境区间

第一轮尚无历史：

```python
delta_observation = turn.prompt_tokens
```

因此：

```text
F₁ = P₁ + C₁
```

第二轮及以后，先检查当前 prompt 是否以前面拼好的完整轨迹为前缀：

```python
elif turn.prompt_tokens[: len(full_tokens)] == full_tokens:
    delta_observation = turn.prompt_tokens[len(full_tokens):]
else:
    raise ValueError(...)
```

若进入第 `i` 轮前已有 `Fᵢ₋₁`：

```text
Pᵢ = Fᵢ₋₁ + Dᵢ
Fᵢ = Fᵢ₋₁ + Dᵢ + Cᵢ
```

这个前缀检查守住了 PPO 最重要的事实边界：旧 logprob 必须继续对应 sampler 当时真正生成的 token。若后续 prompt 重编码或改写了旧历史，代码宁可报错，也不静默错位训练。

## 7.6 token 来源如何决定训练 mask

每一轮按来源扩展三个等长列表：

```python
full_tokens.extend(delta_observation)
full_tokens.extend(turn.completion_tokens)

old_logprobs_by_token.extend([0.0] * len(delta_observation))
old_logprobs_by_token.extend(turn.logprobs)

advantages_by_token.extend([0.0] * len(delta_observation))
advantages_by_token.extend(
    [trajectory.advantage] * len(turn.completion_tokens)
)
```

| token 来源 | 是否保留在序列 | old logprob | advantage |
| --- | --- | --- | --- |
| 初始 system/user prompt | 是 | `0` 占位 | `0` |
| sampler 返回的 assistant token | 是 | 真实逐 token 值 | 轨迹 advantage `A` |
| 模板补齐的 closing | 是 | `0` 占位 | `0` |
| tool wrapper 与 observation | 是 | `0` 占位 | `0` |
| 下一轮 assistant prefix | 是 | `0` 占位 | `0` |

这里没有单独的 Boolean loss mask，而是用 `advantages == 0` 隐式屏蔽非模型动作。

环境位置的 `logprobs=0` 只是为保持 shape 对齐的占位值，不表示旧策略真的给这些 token 分配了对数概率 0。

### closing token 到底训不训练

不能只看它在语义上是不是 closing，要看它来自哪里：

```text
sampler 已生成的 closing token → 模型动作，参与训练
模板为续接而补上的 closing   → 环境增量，不参与训练
```

这正是第三章 `build_next_prompt()` 处理 token overlap 的训练意义。

## 7.7 为什么四个数组都要统一右移

拼接完成后，同一位置保存：

```text
full_tokens[j]
old_logprobs_by_token[j]
advantages_by_token[j]
```

但因果语言模型在位置 `j` 预测的是下一个 token，所以源码统一右移：

```python
input_tokens = full_tokens[:-1]
target_tokens = full_tokens[1:]
old_logprobs = old_logprobs_by_token[1:]
advantages = advantages_by_token[1:]
```

不能只右移 input/target 而保留原 metadata；旧 logprob 和 advantage 描述的是目标 token，也必须一起取 `[1:]`。

## 7.8 一个完整的两轮 token 例子

假设：

```python
P1  = [101, 102, 103]
C1  = [201, 202, 90]
LP1 = [-0.1, -0.2, -0.3]

D2  = [91, 301, 302, 303, 104]
C2  = [401, 402]
LP2 = [-0.4, -0.5]

A = 0.75
```

其中：

```text
90       → sampler 生成的 closing 片段
91       → 模板补齐的 closing 片段
301~303  → tool wrapper 与 observation
104      → 下一轮 assistant prefix
```

拼接但尚未右移时：

```python
full_tokens = [
    101, 102, 103,
    201, 202, 90,
    91, 301, 302, 303, 104,
    401, 402,
]

old_logprobs_by_token = [
     0.0,  0.0,  0.0,
    -0.1, -0.2, -0.3,
     0.0,  0.0,  0.0, 0.0, 0.0,
    -0.4, -0.5,
]

advantages_by_token = [
    0.0, 0.0, 0.0,
    0.75, 0.75, 0.75,
    0.0, 0.0, 0.0, 0.0, 0.0,
    0.75, 0.75,
]
```

右移后的关键位置：

| model input 尾 token | target token | old logprob | advantage | 含义 |
| ---: | ---: | ---: | ---: | --- |
| `103` | `201` | `-0.1` | `0.75` | 第一轮首个模型动作 |
| `90` | `91` | `0` | `0` | 预测模板补齐 token，被 mask |
| `104` | `401` | `-0.4` | `0.75` | 第二轮首个模型动作 |
| `401` | `402` | `-0.5` | `0.75` | 第二轮后续模型动作 |

本例真正参与 loss 的位置数为：

```text
len(C1) + len(C2) = 3 + 2 = 5
```

`datum_loss_token_count()` 正是通过统计 `advantages != 0` 得到这个数。

## 7.9 最终 `Datum` 的字段与校验

设拼接后总长为 `T`，右移后长度为 `L=T-1`：

| 字段 | dtype | 逻辑 shape | 含义 |
| --- | --- | --- | --- |
| `model_input` | 由 `ModelInput.from_ints()` 接收整数 ID | `[L]` | 模型输入 |
| `target_tokens` | `int64` | `[L]` | 下一 token 标签 |
| `logprobs` | `float32` | `[L]` | 旧策略对目标 token 的 logprob |
| `advantages` | `float32` | `[L]` | 轨迹信号兼 loss mask |

提交时 PyTRIO 会把一批样本 padding 成二维矩形，并生成 attention mask；`build_datum()` 本身只创建一维单样本。

主循环调用 `forward_backward()` 时没有传 `auto_shift`，使用默认值 `False`。因此这里的手工右移就是唯一一次右移，不会被 trainer 再移一遍。

它还检查：

```text
轨迹至少有一个 turn
每轮 completion 与 logprob 等长
后续 prompt 是旧轨迹的前缀扩展
整条轨迹至少有一个 assistant token
拼接前后所有数组等长
右移后 input 长度不超过 8192
```

`build_training_datums()` 只保留：

```text
trajectory.advantage != 0
且至少一轮 completion 非空
```

因此同题组若全对或全错，所有 advantage 都是 0，整组不会发送给 trainer。若整个 rollout batch 都如此，本 step 会跳过更新。

## 7.10 为什么需要动态 micro-batch

不同轨迹长度差异很大，而同一次远端请求要 padding 到批内最长序列。源码用下面的矩形面积作为容量代理：

```text
padded token positions = batch_size × max_sequence_length
```

`pack_micro_batches()` 是 first-fit decreasing：

```text
1. 按 num_tokens 从长到短排序
2. 对每条 Datum，从第一个已有 batch 开始试放
3. 同时满足：条数 ≤ 32，试放后的矩形面积 ≤ 64,000
4. 放入第一个可容纳的 batch；都放不下则新开一个
```

长度为 8192 时，一个 batch 最多放：

```text
floor(64,000 / 8192) = 7 条
```

长度为 2000 时，面积本可容纳 32 条，恰好又被条数上限 32 卡住。

要注意，64,000 不是有效 token 总和、显存字节或 attention FLOPs，只是 padding 矩形大小的简单代理。这个贪心策略也不保证全局 padding 最少。

例如长度：

```text
[8000, 7800, 3000 × 10]
```

源码可能装成：

```text
batch 1: [8000, 7800, 3000 × 6] → 8 × 8000 = 64,000
batch 2: [3000 × 4]             → 4 × 3000 = 12,000
```

总 padding 面积为 76,000。若按相近长度分组则只需 46,000，说明“首个能放就放”更偏向尽快填箱，而非最小化 padding。

## 7.11 为什么每个 micro-batch 要乘 `nₖ/N`

假设第 `k` 个 micro-batch 有 `nₖ` 条，整个 rollout 返回 `N` 条轨迹。每次 `forward_backward()` 在自己的逻辑 batch 内做归一化；若直接累计不同大小批次，小 batch 会获得过大权重。

源码通过缩放 advantage 修正：

```python
micro_batch_weight = len(micro_batch) / total_samples
advantages = advantages * micro_batch_weight
```

即：

```text
cₖ = nₖ / N
```

因为正数缩放不改变 PPO ratio 的 clip 分支，而 loss 对 advantage 线性，所以梯度满足：

```text
Σₖ [nₖ/N × (1/nₖ) Σᵢ∈Bₖ ∇ℓᵢ]
= (1/N) Σᵢ ∇ℓᵢ
```

一个最小数值例子。三条样本损失为 `[2, 4, 6]`，拆成 `2+1`：

```text
batch 1 权重 2/3：mean([2,4] × 2/3) = 2
batch 2 权重 1/3：mean([6]   × 1/3) = 2
累计 = 4 = mean([2,4,6])
```

若不缩放，累计会变成 `3+6=9`。

### 为什么分母是全部 trajectories

主循环传入：

```python
total_samples=len(trajectories)
```

而不是 `len(datums)`。设 rollout 有 `R` 条轨迹，其中只有 `D` 条 advantage 非零：

```text
当前目标 = (非零 Datum 的 loss 总和 + 被过滤零项) / R
```

所以它保持的是“所有 rollout 轨迹均值”。如果改用 `D`，得到的会是“仅有效 Datum 均值”，梯度方向相同，但当前实现的尺度会再乘 `D/R`；退化组越多，实际更新越弱。

## 7.12 一次 step 为何只有一次参数更新

训练段的真实时序是：

```python
for micro_batch in micro_batches:
    training_client.forward_backward(...).result()

if micro_batches:
    training_client.optim_step(adam_params).result()
```

因此：

```text
K 个 micro-batch
→ K 次 forward/backward，梯度累积
→ 1 次 Adam optim_step，参数更新
```

micro-batch 只是把一个逻辑 rollout batch 拆开计算，不是 `K` 个训练 step，也不会在中途改变策略参数。

PPO 配置只显式给出 clipped surrogate 使用的概率比区间：

```text
rₜ = exp(log π_new - log π_old)
clip(rₜ, 0.8, 1.28)
```

这对应下侧最多偏离 `0.2`、上侧最多偏离 `0.28` 的非对称 clip。源码没有在这里实现多 PPO epoch、KL 惩罚或 entropy bonus。

这里的两个名字并不冲突：同题分组和中心化 advantage 是 GRPO 风格，真正交给远端 trainer 的 loss 名称则是 `"ppo"`。

## 7.13 `main()` 启动时创建什么

训练开始前依次构造：

```text
shuffled_examples(data, seed)
  → 固定打乱后的 MathExample 列表

ServiceClient
  → create_lora_training_client(base_model, rank, seed)
  → training_client

training_client.get_tokenizer()
  → rollout 与训练共同使用的 tokenizer

LocalPythonSandbox
RolloutConfig
AdamParams
SwanLab run
```

tokenizer 从训练客户端取得很重要：prompt、sampler token、旧 logprob 与训练 target 必须属于同一套词表和编码规则。

源码显式设置 Adam 的：

```text
learning_rate = 4e-5
beta1 = 0.9
beta2 = 0.95
```

在当前锁定的 PyTRIO 0.2.6 中，其余默认值是 `eps=1e-12`、`weight_decay=0`、`grad_clip_norm=0`，即没有权重衰减和梯度裁剪。

## 7.14 当前策略、旧策略和更新后的策略

令第 `s` 步开头的 LoRA 参数为 `θₛ`：

```text
training_client 持有 θₛ
        ↓ save_weights_and_get_sampling_client()
sampling_client 固定为 θₛ 的采样快照
        ↓
本 step 的全部 rollout 和 old logprobs 来自 θₛ
        ↓
所有 micro-batch 在 θₛ 上累计梯度
        ↓ optim_step() 一次
training_client 变成 θₛ₊₁
        ↓
周期 checkpoint 保存 θₛ₊₁
        ↓
下一 step 才导出 θₛ₊₁ 的 sampler
```

每一步都会重新导出 sampler。它是当步策略快照，不是会随 training client 自动同步的长期对象。

## 7.15 一个完整 training step

主循环可压缩为：

```text
1. take_batch() 循环取 questions_per_batch 道题
2. 从当前 LoRA 导出 sampling client
3. rollout_batch() 生成 Q × G 条轨迹并计算 reward/advantage
4. 过滤零 advantage 轨迹并构造 Datum
5. 按 padding 预算动态装箱
6. 对每个 micro-batch 做 PPO forward/backward
7. 若有 micro-batch，只做一次 Adam update
8. 汇总指标
9. 按周期保存 checkpoint
10. 记录 SwanLab 和终端进度
```

默认 `questions_per_batch=8`、`group_size=8` 时，正常情况下一个 step 先得到 64 条轨迹。

训练集只在程序启动时固定打乱一次；之后：

```python
start = step * questions_per_batch
take_batch(...)
```

通过取模循环回绕。这里没有 epoch 结束重洗牌；若数据量小于 batch size，同一道题甚至可能在一个 step 内重复出现。

## 7.16 什么时候跳过更新

若 `micro_batches` 为空：

```text
不调用 forward_backward
不调用 optim_step
train/update_skipped = 1
```

最常见原因是所有题组都退化：同题候选全对或全错，组内 advantage 全为零。

但这个 step 仍然：

```text
计入进度
记录 rollout 指标
可能按 save_every 保存一份参数未变化的 checkpoint
```

长度错位、Datum 超过 8192 等属于异常，会直接终止训练，而不是“安全跳过”。

## 7.17 checkpoint 为什么保存两份

```python
save_state(name=f"{name}-state")
save_weights_for_sampler(name=f"{name}-weights")
```

| 产物 | 用途 |
| --- | --- |
| `*-state` | 训练断点所需的完整 state |
| `*-weights` | sampler / `eval.py` 可加载的推理权重 |

周期 checkpoint 位于可选的 `optim_step()` 之后，因此 `step-25` 保存的是第 25 个外层 step 结束后的参数；若某些 step 跳过更新，它不一定代表第 25 次实际参数更新。

`save_every=0` 只关闭周期保存；训练正常结束仍会保存 `final`。不过当前 `train.py` 没有 resume 参数或加载 state 的代码，所以虽然保存了断点，本脚本本身还不能直接续训。

`finally` 只保证 `run.finish()`：若训练中途异常，不会自动保存紧急 final checkpoint。

## 7.18 指标应该怎样读

### rollout 与 reward

| 指标 | 实际语义 |
| --- | --- |
| `reward/mean` | 所有轨迹的 `±1` 平均值 |
| `reward/correct` | 数学答案正确率 |
| `reward/format` | 找到合法 boxed 的比例 |
| `rollout/turns` | 每轨迹 assistant turn 平均数 |
| `rollout/code_calls` | 成功接入历史的 tool observation 平均数 |
| `rollout/trajectory_tokens` | 最后一轮 prompt 加 completion 的平均长度 |
| `rollout/degenerate_group_rate` | advantage 全零的问题组比例 |

`rollout/valid_tool_call_rate` 的分子是成功接入历史的 `code_calls`，分母是 assistant 文本中出现 `<tool_call>` 的 turn 数。因此它更接近“工具协议被接受的比例”，不是 Python 执行成功率。

### 训练张量

```text
train/tokens_per_rollout_batch
→ 所有 Datum 的 input token；包含被 mask 的 prompt/observation

train/loss_tokens_per_rollout_batch
→ advantages != 0 的目标 token 数

train/padded_tokens_per_rollout_batch
→ 各手工 micro-batch 的 padding 矩形面积之和
```

### sandbox 与 trainer 指标的两个口径提醒

同一个 sandbox 对象跨 step 复用，其 `SandboxStats` 从不清零，所以日志中的成功率、超时率和平均延迟是“从训练开始到当前”的累计值，不是单步值。

`merge_trainer_metrics()` 对预加权后的 `loss_mean` 求和，其他数值指标却按 micro-batch 简单平均。在 PyTRIO 0.2.6 中，单次返回的：

```text
loss_mean = loss_sum / token_count
```

所以合并后的展示 loss 可用于观察趋势，却不应武断解释成严格的全局 sample mean 或全局 token mean；不同 micro-batch 长度差异越大，口径差异越明显。`loss_sum`、`token_count` 被简单平均后也不再具有整步“总量”语义。这影响诊断指标解释，不改变前面的梯度累计逻辑。

## 7.19 关键边界与源码疑点

### 边界 1：rollout 上限与训练上限可以冲突

`--max-trajectory-tokens` 可设为大于 8192，但 `build_datum()` 固定拒绝右移后超过 8192 的输入。轨迹可能先成功生成，再在训练转换阶段报错。

### 边界 2：最后一个 observation 未必写入 Datum

observation 只有在下一轮 `turn.prompt_tokens` 中才被 `build_datum()` 发现。若工具返回后立刻因预算耗尽而没有下一次 assistant turn，最后的 observation 不会进入 Datum。

这通常不改变梯度：它本应被 mask，且后面已经没有依赖它的模型动作。

### 边界 3：采样 seed 没有包含 global step

rollout seed 由固定 base seed、当前 batch 内 index 和 turn 数构成，没有加入训练 step。题目再次出现在相同槽位时会复用同一 seed；若策略也未变化，例如连续跳过更新，可能重复同样的采样。

### 边界 4：多数 CLI 数值没有范围校验

例如：

```text
group_size=1          → advantage 恒为 0
max_steps<=0          → 不训练，仍保存初始 final
questions_per_batch=0 → 每步空 batch并持续跳过
```

非法负值也多由后续模块或远端服务才发现。

### 边界 5：指标可能低估“消失的问题”

若初始 prompt 已耗尽 rollout 预算，该题不会分叉成轨迹。`degenerate_group_rate` 的分母仍是 batch 题数，但该题没有进入分子，因而可能低估异常组。

### 边界 6：保存不是事务操作

源码先保存 state，再保存 sampler weights。第二步失败时可能只留下 state；checkpoint 又发生在 SwanLab log 之前，因此保存异常也会让当前 step 的指标没有写入。

## 7.20 本章对象账本

| 阶段 | 输入 | 输出 |
| --- | --- | --- |
| 轨迹过滤 | `list[Trajectory]` | 非零 advantage 的轨迹 |
| token 合并 | 多个 `AssistantTurn` | 一条连续 `full_tokens` |
| loss 对齐 | token / old logprob / advantage | 右移后的 `trio.Datum` |
| 动态装箱 | `list[TrainingDatum]` | `list[list[TrainingDatum]]` |
| 批权重修正 | 第 `k` 批与全局 `N` | advantage 乘 `nₖ/N` 的 Datum |
| 梯度阶段 | K 个 micro-batch | 累计梯度 |
| 优化阶段 | 累计梯度 | 更新后的 LoRA 参数 |
| 保存阶段 | training client | state 与 sampler weights |

## 7.21 本章小结

用一句话概括 `train.py`：

> 它把每条多轮轨迹按真实 token 前缀合并成一个右移 PPO 样本，只让 sampler 生成的 assistant token 携带旧 logprob 和 advantage；随后按 padding 矩形动态拆批，用 `nₖ/N` 保持全 rollout 样本的梯度均值，累计完所有 backward 后只更新一次 LoRA，并记录指标与双份 checkpoint。

读完本章应能回答：

1. 为什么一条多轮轨迹只产生一个 Datum？
2. `delta_observation` 为什么不只是 stdout？
3. prompt 和 observation 为什么保留 token、却把 advantage 置零？
4. logprob 与 advantage 为什么也必须一起右移？
5. closing token 是否参与训练由什么决定？
6. 动态装箱限制的是有效 token 总和吗？
7. 为什么 micro-batch advantage 要乘 `nₖ/N`？
8. 为什么 `N` 使用全部 trajectories，而不是 datums？
9. K 次 `forward_backward()` 为什么仍只算一个训练 step？
10. 周期 checkpoint 保存的是 rollout 前还是更新后的参数？
11. 哪些情况会跳过 `optim_step()`？
12. 哪些指标是累计口径，哪些 trainer 指标不能当作整步总量？
