---
title: ReTool 源码精读04
publishDate: 2026-08-12
description: '第四章：`sandbox.py`——代码调用如何变成 observation'
tags:
  - 笔记
---

# 第四章：`sandbox.py`——代码调用如何变成 observation

源码位置：`https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/05-retool/sandbox.py`

## 4.1 先抓住本章主线

上一章的 `parse_assistant()` 从模型输出中提取出 Python 字符串：

```python
code = "print(2 + 3)"
```

`sandbox.py` 把它变成下一轮模型可见的环境反馈：

```text
Python code: str
    ↓ LocalPythonSandbox.arun_code()
线程池中的同步 run_code()
    ↓ subprocess.Popen()
全新 Python 子进程
    ↓ stdout / stderr / timeout
ExecResult
    ↓ format_tool_content()
tool content: str
    ↓ protocol.tool_message()
role="tool" 的结构化消息
    ↓ protocol.build_next_prompt()
下一轮 observation tokens
```

本章结束时，数据第一次经历：

```text
模型动作中的代码文本
    → 外部环境执行
    → 环境 observation 文本
```

需要牢牢记住：代码是模型生成的动作；执行结果是环境返回的 observation。二者在训练时的 mask 不同。

## 4.2 “沙箱”在这里具体指什么

当前类名是：

```python
LocalPythonSandbox
```

它提供的主要能力是：

- 每次调用创建一个新 Python 子进程；
- 限制同时运行的子进程数量；
- 设置 wall-clock 超时和 CPU 时间上限；
- 超时时杀死进程组；
- 把数值库线程数限制为 1；
- 控制返回给模型的 stdout/stderr 长度；
- 消毒可能破坏 chat template 的特殊文本。

但它不是容器、虚拟机、seccomp 或受限操作系统账户意义上的强安全沙箱。源码没有强制禁止：

- 读取或写入本地文件；
- 访问网络；
- 读取继承的环境变量；
- 创建额外进程；
- 调用任意可用系统 API；
- 占用大量内存；
- 在执行期间向临时输出文件写入大量数据。

因此更准确的定位是：

> 它是面向受控数学实验的“带资源保护的本地 Python 执行器”，不能直接当作执行任意恶意代码的安全边界。

system prompt 虽然要求模型“不读写文件、少用资源”，但那是行为指令，不是操作系统强制限制。

## 4.3 三个基础限制常量

```python
DEFAULT_TIMEOUT = 30.0
MAX_OUTPUT_CHARS = 4096
READ_CAP_BYTES = MAX_OUTPUT_CHARS * 4
```

### `DEFAULT_TIMEOUT`

一次代码调用默认最多等待 30 秒 wall-clock 时间。wall-clock 包括真实世界经过的时间：CPU 计算、阻塞、休眠等都计算在内。

### `MAX_OUTPUT_CHARS`

stdout 和 stderr 读回后，各自最多保留大约 4096 个字符的尾部。

### `READ_CAP_BYTES`

先从临时文件末尾读取最多：

```text
4096 × 4 = 16384 bytes
```

乘以 4 是为 UTF-8 多字节字符留余量。随后再按 Python 字符数截断到 4096。

这里有两层不同单位：

```text
文件读取上限 → bytes
返回文本上限 → Python characters
```

它们都不是 tokenizer token 数。rollout 后面还会做一次 token 级截断。

## 4.4 `_BOOTSTRAP`：真正传给子进程的引导代码

源码模板：

```python
_BOOTSTRAP = (
    "import resource\n"
    "resource.setrlimit(resource.RLIMIT_CPU, ({cpu}, {cpu}))\n"
    "exec(compile({code}, '<sandbox>', 'exec'))\n"
)
```

假设模型代码是：

```python
print(2 + 3)
```

默认 timeout 为 30 秒，父进程会生成类似的引导代码：

```python
import resource
resource.setrlimit(resource.RLIMIT_CPU, (35, 35))
exec(compile('print(2 + 3)', '<sandbox>', 'exec'))
```

CPU 限额计算为：

```python
int(self.timeout) + 5
```

默认得到 35 秒，比 30 秒 wall-clock 超时多 5 秒，主要作为内核级兜底。通常父进程的 wall-clock 超时会先触发。

### `compile(..., '<sandbox>', 'exec')`

- `exec` 模式允许执行多行语句；
- `'<sandbox>'` 是虚拟文件名，异常 traceback 会把代码位置显示为 `<sandbox>`；
- 代码没有写入 `.py` 文件。

### 为什么使用 `repr(code)`

构造引导代码时：

```python
bootstrap = _BOOTSTRAP.format(
    cpu=int(self.timeout) + 5,
    code=repr(code),
)
```

`repr()` 把换行、引号等字符变成合法的 Python 字符串字面量。例如：

```python
code = "print('hello')\nprint(2 + 3)"
```

`repr(code)` 类似：

```text
"print('hello')\nprint(2 + 3)"
```

这样模型代码被作为 `compile()` 的字符串参数嵌入，而不是拼成 shell 命令。

需要准确理解“无注入面”的范围：它避免的是 shell argv 拼接注入；模型代码本来就会被 `exec()` 有意执行，因此这不等于代码本身安全。

## 4.5 子进程启动命令

核心代码：

```python
process = subprocess.Popen(
    [self.python, "-B", "-c", bootstrap],
    stdout=stdout_file,
    stderr=stderr_file,
    env=env,
    start_new_session=True,
)
```

参数列表可以理解为：

```text
当前 Python 解释器
    -B
    -c
    <bootstrap 字符串>
```

### list 形式 argv

使用列表而不是 shell 命令字符串，不经过 shell 解析。代码里的引号、分号和重定向字符不会被父进程 shell 当作命令语法解释。

### `-B`

禁止 Python 写入 `.pyc` 字节码缓存。它只阻止自动生成字节码缓存，并不阻止模型代码主动写文件。

### `-c`

让 Python 直接执行后面的引导字符串，不需要创建临时脚本文件。

### `start_new_session=True`

在 POSIX 系统上让子进程成为新 session/进程组的起点。这样超时时可以按进程组杀死，而不只是杀最外层 Python 进程。

## 4.6 每次调用为什么“变量无状态”

每次 `run_code()` 都会重新执行一次 `subprocess.Popen()`：

```text
调用 1 → Python 子进程 A → 结束
调用 2 → Python 子进程 B → 结束
```

例如第一次执行：

```python
x = 5
print(x)
```

第二次执行：

```python
print(x)
```

第二个进程没有第一个进程的内存变量，因此会得到 `NameError`。

不过“无状态”主要指进程内存和 Python 变量。因为源码没有隔离文件系统，若第一段代码主动写文件，第二个进程理论上仍可能读取它。因此“文件也不持久化”目前主要依赖 system prompt，而不是实现强制保证。

## 4.7 `_CHILD_ENV_OVERRIDES`：限制数值库线程

源码设置：

```python
_CHILD_ENV_OVERRIDES = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
```

创建子进程环境时：

```python
env = {**os.environ, **_CHILD_ENV_OVERRIDES}
```

它先复制父进程的全部环境变量，再用上述四个值覆盖同名变量。

目的不是限制 Python 线程语法，而是避免 NumPy/OpenBLAS/MKL 等数值库在每个沙箱进程里自动启动大量计算线程。假设同时执行 8 段代码，而每段 NumPy 又启动 16 个 BLAS 线程，就可能瞬间形成 128 个计算线程。

同时也要注意：因为继承了 `os.environ`，父进程中的其他环境变量对子进程可见。这进一步说明当前实现不是秘密隔离边界。

## 4.8 `ExecResult`：一次调用的不可变结果

定义：

```python
@dataclass(frozen=True)
class ExecResult:
    ok: bool
    stdout: str
    stderr: str
    returncode: int | None
    timed_out: bool
    latency: float
```

字段含义：

| 字段 | 含义 |
| --- | --- |
| `ok` | 没有超时并且退出码为 0 |
| `stdout` | 截断后的标准输出 |
| `stderr` | 截断后的标准错误 |
| `returncode` | 正常回收时的进程退出码；超时路径设为 `None` |
| `timed_out` | 是否由 wall-clock 超时触发杀进程 |
| `latency` | 本次 `run_code()` 从计时开始到结果构造的耗时 |

`frozen=True` 表示结果生成后不应被下游修改。

### 三种典型结果

成功执行：

```python
ExecResult(True, "5\n", "", 0, False, 0.03)
```

运行时错误：

```python
ExecResult(False, "", "Traceback ...", 1, False, 0.03)
```

超时：

```python
ExecResult(False, "", "", None, True, 30.0)
```

## 4.9 `SandboxStats`：跨调用累计指标

与不可变的单次结果不同，`SandboxStats` 是持续变化的累计器：

```text
calls
successes
errors
timeouts
latency_total
```

分类规则：

```text
timed_out=True                  → timeouts += 1
没有超时且 returncode == 0     → successes += 1
其他非零退出                   → errors += 1
```

`metrics()` 转成：

```python
{
    "sandbox/success_rate": successes / calls,
    "sandbox/error_rate": errors / calls,
    "sandbox/timeout_rate": timeouts / calls,
    "sandbox/latency": latency_total / calls,
}
```

`max(self.calls, 1)` 用来避免还没有任何调用时除以 0。

这些指标最终和 rollout、reward、trainer 指标一起记录到 SwanLab。它们描述工具环境是否健康，不直接作为模型 reward。

## 4.10 为什么 stdout/stderr 不使用 `PIPE`

常见写法是：

```python
stdout=subprocess.PIPE
stderr=subprocess.PIPE
```

但模型可能生成：

```python
while True:
    print("x" * 1000000)
```

若父进程没有及时消费 PIPE，缓冲区可能塞满，子进程阻塞；若一次把大量输出读进内存，又可能撑爆父进程内存。

当前代码改用：

```python
tempfile.TemporaryFile()
```

stdout 和 stderr 各写进一个匿名临时文件。进程结束后，父进程只从文件尾部读取有限字节。

这样保护了父进程内存，但不代表执行期间的临时文件大小被硬限制。输出洪水仍可能在超时前消耗临时磁盘空间，只是不会全部进入 Python 内存。

## 4.11 `_read_capped()`：只读文件尾部

流程：

```python
file.seek(0, os.SEEK_END)
size = file.tell()
file.seek(max(0, size - limit))
return file.read().decode("utf-8", errors="replace")
```

可以理解为：

```text
定位文件末尾
→ 得到文件大小
→ 向前移动最多 READ_CAP_BYTES
→ 只读取最后一段
→ 按 UTF-8 解码
```

为什么保留尾部而不是头部？

- 正常程序通常在最后打印最终结果；
- traceback 的异常类型和最终报错信息位于末尾；
- 前面可能只是大量中间日志。

若读取起点恰好落在 UTF-8 多字节字符中间，`errors="replace"` 会用替代字符继续解码，而不是让整个工具调用失败。

## 4.12 `_truncate_tail()`：按字符再次截断

```python
if len(text) <= limit:
    return text
omitted = len(text) - limit
return f"[... truncated {omitted} chars ...]\n{text[-limit:]}"
```

例如输出有 10,000 个字符、limit 为 4,096，则模型看到：

```text
[... truncated 5904 chars ...]
<最后 4096 个字符>
```

提示行告诉模型前面内容被省略了。

严格说，返回字符串总长度会略大于 4096，因为还额外包含 truncation 提示；4096 是保留的原始尾部字符数。

## 4.13 `sanitize_tool_content()`：防止 observation 破坏模板

假设模型代码打印：

```text
<|im_end|>
```

如果 chat template 把 tool content 原样嵌入，这段文本可能被 tokenizer 识别成真正的消息结束标记，提前破坏 observation 边界。

源码把几个敏感标记替换成不再完全匹配的普通文本：

```text
<|im_start|>       → < im_start >
<|im_end|>         → < im_end >
<tool_response>    → < tool_response >
</tool_response>   → < /tool_response>
```

这不是为了隐藏错误内容，而是保证 stdout/stderr 只能作为 tool 消息正文，不能伪装成 chat template 的结构控制标记。

消毒发生在 `format_tool_content()`，所以 `ExecResult.stdout/stderr` 仍保存截断后的原始文本；真正回喂模型时才替换。

## 4.14 `LocalPythonSandbox.__init__()`

初始化保存：

```python
self.timeout = timeout
self.python = python
self.stats = SandboxStats()
self._semaphore = threading.BoundedSemaphore(max_workers)
```

默认 Python 是 `sys.executable`，也就是运行训练脚本的同一个 Python 环境，因而子进程可以导入该环境中已安装的 NumPy 等库。

### 为什么是 `threading.BoundedSemaphore`

训练中的多个轨迹可能在同一 round 同时请求执行代码。信号量确保最多只有 `max_workers` 个 `run_code()` 进入子进程执行区域。

项目的同步训练入口每个 step 都会通过 `asyncio.run()` 创建新的事件循环，而同一个 sandbox 对象跨 step 复用。若在对象里长期保存一个 `asyncio.Semaphore`，它可能绑定到旧事件循环；`threading.BoundedSemaphore` 不绑定 event loop，可以跨这些调用复用。

`BoundedSemaphore` 还会在错误地释放超过初始容量时抛异常。当前代码通过 `with self._semaphore` 自动获取和释放。

## 4.15 `run_code()` 完整执行过程

以 `code = "print(2 + 3)"` 为例。

### 第一步：计时和计数

```python
started = time.perf_counter()
self.stats.calls += 1
```

### 第二步：构造引导代码和环境变量

```python
bootstrap = _BOOTSTRAP.format(...)
env = {**os.environ, **_CHILD_ENV_OVERRIDES}
```

### 第三步：等待并发槽位

```python
with self._semaphore:
```

若已有 `max_workers` 个调用正在执行，新调用会等待。计时发生在获取 semaphore 之前，因此 `latency` 实际包含排队时间，不只是子进程运行时间。

### 第四步：创建 stdout/stderr 临时文件

```python
with TemporaryFile() as stdout_file, TemporaryFile() as stderr_file:
```

退出 `with` 后文件自动关闭和清理。

### 第五步：启动全新 Python 子进程

```python
process = subprocess.Popen(...)
```

子进程执行引导代码，引导代码再 `compile + exec` 模型代码。

### 第六步：等待正常退出或超时

```python
try:
    returncode = process.wait(timeout=self.timeout)
except subprocess.TimeoutExpired:
    ...
```

正常退出时，退出码 0 表示成功，非 0 通常表示语法错误、未捕获异常或其他失败。

### 第七步：超时时杀进程组

```python
os.killpg(process.pid, signal.SIGKILL)
returncode = None
process.wait()
```

`killpg` 杀死整个进程组，避免只杀外层 Python，而模型代码派生的子进程继续运行。再次 `wait()` 用于回收已终止进程，避免僵尸进程。

进程组清理只发生在超时路径；正常退出路径不会额外清理模型代码可能留下的后台进程。

### 第八步：读取并截断输出

```python
stdout = _truncate_tail(_read_capped(stdout_file))
stderr = _truncate_tail(_read_capped(stderr_file))
```

stdout 和 stderr 独立处理。

### 第九步：构造结果

```python
ok = not timed_out and returncode == 0
```

代码再更新累计统计，最后返回 `ExecResult`。

## 4.16 三种代码调用怎样变成 observation

### 成功并打印结果

模型代码：

```python
print(2 + 3)
```

执行结果大致为：

```python
ExecResult(
    ok=True,
    stdout="5\n",
    stderr="",
    returncode=0,
    timed_out=False,
    latency=0.03,
)
```

`format_tool_content()` 返回字符串 `"5\n"`，下一轮模型就能利用这个结果继续推理。

### 代码抛出异常

模型代码：

```python
print(1 / 0)
```

子进程以非零退出码结束，stderr 含 traceback。格式化时把 traceback 回喂模型，例如：

```text
Traceback (most recent call last):
  File "<sandbox>", line 1, in <module>
ZeroDivisionError: division by zero
```

模型可以在下一轮分析错误并修改代码。ReTool 的自我纠错不是沙箱自动修复，而是模型看到错误 observation 后产生新的 assistant 动作。

### 代码超时

模型代码：

```python
while True:
    pass
```

超时后 `format_tool_content()` 返回统一文本：

```text
Error: execution timed out after 30 seconds.
```

模型可以据此缩小枚举范围或改用更高效算法。

## 4.17 `format_tool_content()` 的优先级

源码顺序：

```text
timed_out
    → 返回统一超时信息

否则 ok
    → 返回消毒后的 stdout

否则 stderr 非空
    → 返回消毒后的 stderr

否则
    → 返回通用退出码错误
```

因此：

- 成功路径只回 stdout，即使 stderr 中有 warning，也不会回给模型；
- 失败路径优先回 stderr，即使失败前 stdout 有部分输出，也不会回 stdout；
- 成功但没有 `print()` 时返回空字符串；
- 特殊模板标记只在回喂前消毒。

这是一个清晰但有取舍的“单通道 observation”策略。

## 4.18 `ensure_trailing_print()`：已定义但当前没有接入执行路径

函数意图是把最后一个非空行：

```python
2 + 3
```

改成：

```python
print(2 + 3)
```

算法会从代码最后一行向前找第一个非空行；如果该行不是以字面量 `print` 开头，就外包一层 `print(...)`。

但是当前源码中的 `run_code()` 使用：

```python
code=repr(code)
```

并没有调用：

```python
ensure_trailing_print(code)
```

全项目搜索也没有发现其他调用者。因此按当前真实路径：

```python
run_code("2 + 3")
```

进程会成功退出，但 stdout 为空；只有：

```python
run_code("print(2 + 3)")
```

才会返回 `5`。

这与模块头部“auto-print 补全已验证”的说明不完全一致，是后续本地测试与兼容性任务应核实的源码疑点。本章只记录，不修改实现。

即使未来接入，该函数也只适合最后一行是表达式的简单情况：最后一行若是赋值、控制语句或带缩进代码，直接包 `print(...)` 可能产生非法语法。

## 4.19 `arun_code()`：异步外壳，内部仍是同步子进程

```python
async def arun_code(self, code):
    return await asyncio.to_thread(self.run_code, code)
```

`run_code()` 会阻塞等待子进程，不能直接放在 event loop 主线程中，否则一段 30 秒代码会让其他异步任务也无法推进。

`asyncio.to_thread()` 把同步函数送到线程池：

```text
event loop
    ├─ arun_code(A) → 工作线程 A → 子进程 A
    ├─ arun_code(B) → 工作线程 B → 子进程 B
    └─ arun_code(C) → 工作线程 C → 等待 semaphore
```

rollout 使用：

```python
await asyncio.gather(
    *(sandbox.arun_code(p.code) for p in pendings)
)
```

同一 round 中多条轨迹的代码可以并发执行，而 `BoundedSemaphore` 把真正同时运行的子进程限制在 `max_workers` 以内。

并发不会让一次 Python 调用共享变量：每个工作线程最终仍会启动独立子进程。

## 4.20 两层输出预算

sandbox 的输出字符限制不是最终进入模型的唯一限制。

第一层在 `sandbox.py`：

```text
stdout/stderr
→ 最多读取尾部约 16 KB
→ 最多保留尾部 4096 字符
```

第二层在 `rollout.py` 的 `fit_tool_content()`：

```text
tool content
→ tokenizer 统计 token 数
→ 默认最多 512 tool-response tokens
→ 还必须放得进整条 8192-token 轨迹预算
```

因此：

```text
字符级保护 → 防止父进程接收巨大文本
token 级保护 → 防止 observation 撑爆模型上下文
```

如果 tool content 太长，rollout 也会优先保留尾部；若反复缩短到很小仍放不进总轨迹预算，轨迹会直接结束。

## 4.21 observation 回到模型的完整路径

以成功输出 `5\n` 为例：

```text
模型 completion 中的 code
    │ "print(2 + 3)"
    ▼
PendingExecution.code
    ▼
sandbox.arun_code()
    ▼
run_code()
    ▼
Python 子进程 stdout="5\n"
    ▼
ExecResult(ok=True, stdout="5\n", ...)
    ▼
format_tool_content()
    │ "5\n"
    ▼
fit_tool_content()
    │ 字符串满足 token/轨迹预算
    ▼
tool_message(call_id, "5\n")
    ▼
{"role":"tool", "content":"5\n", ...}
    ▼
build_next_prompt()
    ▼
tool observation tokens + 下一轮 assistant 前缀
```

这条路径跨越三个模块：

```text
sandbox.py  → 执行与格式化结果
protocol.py → 包装 tool message、构造 token 增量
rollout.py  → 驱动状态转换和预算检查
```

## 4.22 observation 为什么不参与 loss

模型确实会在下一轮“看到”工具输出，但工具输出不是模型选择的 token。因果关系是：

```text
模型动作：生成 code
环境状态转移：执行 code
环境 observation：stdout / stderr / timeout
模型下一动作：根据 observation 继续生成
```

因此训练时：

```text
code 所在的 assistant completion token
    → 是模型动作，使用轨迹 advantage

tool observation token
    → 是环境输入，advantage = 0
```

observation 必须保留在 `model_input/target_tokens` 的连续序列里，模型才能理解下一轮上下文；但用 0 advantage 把它从 PPO loss 中 mask 掉。具体张量构造将在第七章讲解。

## 4.23 Windows/Unix 平台边界

当前实现依赖 POSIX/Unix 能力：

```python
import resource
resource.RLIMIT_CPU
os.killpg
signal.SIGKILL
start_new_session=True 的进程组语义
```

Windows 原生环境没有同等的 `resource` 和 `os.killpg` 接口。尤其 `_BOOTSTRAP` 在子进程中执行 `import resource` 会直接失败；超时路径的 `os.killpg` 也不可用。

所以当前源码应在 WSL/Linux 路径运行，不能因为父模块在 Windows 上成功 import，就认为代码执行路径可用。

这也是环境路线选择 WSL/Linux 的主要原因之一。

## 4.24 资源限制与安全隔离对照

| 能力 | 当前是否强制实现 | 说明 |
| --- | --- | --- |
| 每次新进程 | 是 | Python 变量不跨调用保留 |
| 并发数限制 | 是 | `BoundedSemaphore` |
| wall-clock 超时 | 是 | `process.wait(timeout=...)` |
| CPU 时间限制 | 是，POSIX | `RLIMIT_CPU` |
| 超时杀进程组 | 是，POSIX | `killpg + SIGKILL` |
| 数值库线程限制 | 是 | 环境变量设置为 1 |
| 返回文本截断 | 是 | 字节读取上限 + 字符尾部截断 |
| 特殊模板标记消毒 | 是 | 四个已知标记替换 |
| 文件系统隔离 | 否 | 代码使用当前 OS 账户权限 |
| 网络隔离 | 否 | 没有禁用网络 |
| 环境变量秘密隔离 | 否 | 继承 `os.environ` |
| 内存硬限制 | 否 | 没有 `RLIMIT_AS` 等 |
| 进程数量限制 | 否 | 没有 `RLIMIT_NPROC` 等 |
| 临时输出文件硬上限 | 否 | 只限制最后读回多少 |

这个表决定了它适合什么场景：受控研究复现、模型主要生成数学计算代码；不适合直接暴露为多租户不可信代码执行服务。

## 4.25 其他源码细节与疑难点

### 统计更新的并发一致性

多个 `arun_code()` 会在线程中同时修改同一个 `SandboxStats`，当前没有显式锁。训练日志通常在整轮 `gather()` 完成后读取，因此实践中大多得到完整计数；若要把它做成更通用的并发组件，应进一步审视计数更新的原子性。

### `latency` 包含排队时间

计时在 semaphore 获取之前开始，所以高并发拥堵时，`sandbox/latency` 同时反映排队和执行耗时。源码注释“从启动到回收”不完全精确。

### 正常退出不清理后台子进程

进程组强杀只在 timeout 分支执行。若模型代码主动启动后台进程后让主 Python 正常退出，当前实现不会在成功路径清理整个组。

### 特殊标记列表不是通用转义系统

`sanitize_tool_content()` 只覆盖当前已知的四个模板标记。如果将来更换 tokenizer/chat template，需要重新检查可能具有结构含义的特殊文本。

## 4.26 本章对象账本

| 阶段 | 对象 | 类型 | 属于模型还是环境 | 是否含 token |
| --- | --- | --- | --- | --- |
| 提取的程序 | `code` | `str` | 模型动作内容 | 否，已解码文本 |
| 待执行记录 | `PendingExecution` | dataclass | rollout 控制对象 | 同时保存原 completion token |
| 子进程引导串 | `bootstrap` | `str` | 环境实现 | 否 |
| 执行结果 | `ExecResult` | frozen dataclass | 环境结果 | 否 |
| 格式化反馈 | `content` | `str` | 环境 observation | 否 |
| 结构化反馈 | `tool_message` | `dict` | 环境 observation | 否 |
| 下一轮增量 | `observation_tokens` | `list[int]` | 环境 observation/context | 是，advantage 应为 0 |

## 4.27 本章小结

用一句话概括 `sandbox.py`：

> 它把模型工具调用中的 Python 字符串放进受并发、时间和返回长度约束的独立子进程执行，将 stdout、traceback 或超时信息整理为下一轮 tool observation；但当前实现不是文件、网络、内存与秘密都隔离的强安全沙箱。

读完本章应能回答：

1. `_BOOTSTRAP` 为什么使用 `repr(code)`、`compile()` 和虚拟文件名 `<sandbox>`？
2. wall-clock timeout 与 `RLIMIT_CPU` 有什么区别？
3. 为什么 stdout/stderr 写临时文件而不是 PIPE？
4. `ExecResult.ok`、`returncode` 和 `timed_out` 是什么关系？
5. traceback 为什么要回喂模型？
6. `arun_code()`、线程池、信号量和子进程各负责什么？
7. 为什么工具结果要进行字符级和 token 级两次截断？
8. 为什么 observation 会进入上下文，却不应该参与 PPO loss？
9. 当前实现为什么不能直接在 Windows 原生环境运行？
10. 当前 `ensure_trailing_print()` 为什么没有产生实际 auto-print 效果？
11. 这个执行器已经限制了哪些资源，又没有隔离哪些能力？
