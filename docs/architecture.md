# Lattice 架构文档

## 总体架构

Lattice 采用五层分层架构，自底向上每层只依赖下层，不存在反向依赖：

```
┌─────────────────────────────────────────────────────────────┐
│  Application Layer                                          │
│  examples/ · 用户代码 · CLI                                  │
├─────────────────────────────────────────────────────────────┤
│  Orchestrator Layer          │  Eval Layer                  │
│  Pipeline · Graph · Supervisor│  Evaluators · EvalRunner     │
├─────────────────────────────────────────────────────────────┤
│  Core Components Layer                                      │
│  Agent · Strategy · Memory · Planner · Tool · ToolKit       │
├─────────────────────────────────────────────────────────────┤
│  LLM Abstraction Layer                                      │
│  LLMProvider Protocol · ProviderRegistry · Config           │
│  Providers: DeepSeek · OpenAI · Anthropic                   │
├─────────────────────────────────────────────────────────────┤
│  Trace Layer                                                │
│  Tracer · Span · SpanEvent · Exporters                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. Trace Layer（追踪层）

最底层，无外部依赖，提供可观测性基础设施。

### 核心类型

| 类 | 职责 |
|---|---|
| `Span` | 一个操作的时间范围，包含 trace_id、span_id、parent_id、attributes、events |
| `SpanEvent` | Span 内的离散事件（如异常） |
| `Tracer` | Span 生命周期管理，通过 `async with tracer.span("name")` 自动嵌套 |

### 数据流

```
Tracer.span("agent.run")
  └── Tracer.span("llm.stream")
  └── Tracer.span("tool.execute")
       └── SpanEvent("exception", error="...")
```

`Tracer` 维护 `_active_span` 栈实现父子关系自动关联。导出支持 JSON（已实现）和 OTLP（预留接口）。

---

## 2. LLM Abstraction Layer（LLM 抽象层）

### LLMProvider Protocol

```python
class LLMProvider(Protocol):
    async def stream(self, model, messages, *, system, tools, temperature, max_tokens, stop, response_format) -> AsyncIterator[StreamEvent]: ...
    async def complete(self, model, messages, **kwargs) -> ModelResponse: ...
```

所有 Provider 实现此 Protocol，无需继承基类。`stream()` 是基础方法，`complete()` 通过消费 stream 实现。

### 类型体系

**Content 联合类型**（消息内容的构建块）：

```
Content = TextContent | ImageContent | ThinkingContent | ToolCall | ToolResult
```

每个类型通过 `type: Literal["..."]` 字段区分，便于模式匹配。

**StreamEvent 联合类型**（流式事件）：

```
StreamEvent = StreamStart | TextDelta | ThinkingDelta
            | ToolCallStart | ToolCallDelta | ToolCallEnd
            | StreamEnd | StreamError
```

`StreamEnd.response` 包含完整的 `ModelResponse`（已解析的 Message + Usage + stop_reason）。

### Provider 实现

| Provider | 模块 | 特点 |
|---|---|---|
| DeepSeek | `providers/deepseek.py` | 继承 OpenAIProvider，配置 DeepSeek API 地址，默认读取 `DEEPSEEK_API_KEY` |
| OpenAI | `providers/openai.py` | 完整实现，支持 tool_calls 增量拼接、stream_options usage、response_format |
| Anthropic | `providers/anthropic.py` | 适配 Anthropic SDK，system 作为顶层参数，content_block 事件映射 |

### ProviderRegistry

```python
registry = ProviderRegistry()  # 全局单例

# 注册
registry.register("deepseek", DeepSeekProvider)

# 解析 model_id
provider, model_name = registry.from_model_id("deepseek:deepseek-v4-pro")
```

`from_model_id()` 解析 `"provider:model"` 格式，按 provider 名查找或创建实例（带缓存）。

### 配置

`lattice.yaml` 通过 `load_config()` 加载，支持 `${ENV_VAR:default}` 环境变量替换。搜索路径：`./lattice.yaml` → `~/.lattice/config.yaml`。

---

## 3. Core Components Layer（核心组件层）

### 3.1 Agent

Agent 是框架核心，封装了 LLM 交互循环：

```python
class Agent:
    def __init__(self, name, *, model, system_prompt, tools, strategy, memory, max_steps): ...
    async def start(self, input): ...   # 初始化消息 + 记忆检索
    async def step(self) -> StepResult: ...  # 执行一步（委托给 Strategy）
    async def run(self, input) -> AgentResult: ...  # 完整循环
    async def run_stream(self, input) -> AsyncIterator[AgentEvent]: ...  # 流式事件
    def clone(self) -> Agent: ...  # 浅拷贝，独立消息历史
```

**Agent 主循环**：

```
start(input)
  ├── 清空消息历史
  ├── memory.retrieve(input) → memory_context
  └── 追加 user 消息

for _ in range(max_steps):
    step()
      ├── 构建 AgentContext（messages + tools + system_prompt + memory_context）
      ├── strategy.step(ctx) → StepResult
      └── 追加新消息到 history
    if StepResult.action == Finish:
        break

memory.store(summary)
return AgentResult
```

### 3.2 Strategy（策略模式）

```python
class Strategy(Protocol):
    async def step(self, ctx: AgentContext) -> StepResult: ...
```

`StepResult` 包含 `messages`（新增的对话消息）和 `action`（`Continue | Finish`）。

#### ReActStrategy

最基础的策略：调用 LLM → 有 tool_calls 则执行工具返回 Continue，否则提取文本返回 Finish。

```
LLM 响应
├── 有 ToolCall → 执行工具 → Continue（继续循环）
└── 纯文本 → Finish（结束）
```

#### PlanAndExecuteStrategy

1. 首次调用：通过 Planner 生成 Plan（DAG）
2. 后续调用：取 `plan.ready_steps()` 中的第一个，构造子上下文执行
3. 步骤失败时触发 `planner.replan()`
4. 全部完成时汇总结果返回 Finish

#### ReflexionStrategy

组合 ReActStrategy，在 Finish 时调用 `ReflectionJudge.judge()` 评估输出质量。未通过则注入反馈消息继续循环，最多反思 `max_reflections` 次。

### 3.3 Tool System（工具系统）

#### @tool 装饰器

```python
@tool
async def my_tool(query: str, limit: int = 10) -> str:
    """工具描述"""
    return "result"
```

内部流程：
1. `inspect.signature()` 提取参数名、类型、默认值
2. `pydantic.create_model()` 动态创建参数模型
3. 包装为 `Tool` 对象，`to_schema()` 生成 JSON Schema

#### ToolKit + 中间件

```python
toolkit = ToolKit(
    name="my_tools",
    tools=[tool_a, tool_b],
    middleware=[RetryMiddleware(max_retries=3), TimeoutMiddleware(timeout=30)],
)
```

中间件采用洋葱模型，执行顺序：

```
Request → Retry → Timeout → 实际执行 → Timeout → Retry → Response
```

| 中间件 | 功能 |
|---|---|
| `RetryMiddleware` | 指数退避重试 |
| `CacheMiddleware` | TTL 缓存，key = 类名 + JSON 参数 |
| `TimeoutMiddleware` | `asyncio.wait_for` 超时控制 |
| `SandboxMiddleware` | 沙箱执行（预留接口） |

#### 内置工具

| 工具 | 功能 |
|---|---|
| `shell` | 异步子进程执行，带超时 |
| `read_file` / `write_file` / `list_dir` | 文件操作 |
| `http_get` / `http_post` | HTTP 请求 |

### 3.4 Memory System（记忆系统）

```python
class Memory(Protocol):
    async def retrieve(self, query: str, top_k: int = 5) -> list[MemoryItem]: ...
    async def store(self, item: MemoryItem) -> None: ...
    async def clear(self) -> None: ...
```

#### 三层记忆

| 层 | 类 | 存储 | 检索 |
|---|---|---|---|
| 工作记忆 | `WorkingMemory` | 内存 FIFO 队列 | 关键词匹配，空查询返回最新 |
| 情景记忆 | `EpisodicMemory` | JSON 文件持久化 | 关键词匹配 / 向量检索（需提供 embedding_fn） |
| 语义记忆 | `SemanticMemory` | VectorStore 后端 | 纯向量检索 |

#### CompositeMemory

加权融合多层记忆：

```python
composite = CompositeMemory([
    (WorkingMemory(), 1.0),      # 权重 1.0
    (EpisodicMemory(...), 0.7),  # 权重 0.7
])
```

`retrieve()` 时并发查询所有子记忆，按 `score * weight` 排序合并。`store()` 只写入第一个（primary）。

#### VectorStore Protocol

```python
class VectorStore(Protocol):
    async def add(self, texts, metadatas) -> list[str]: ...
    async def search(self, query, top_k) -> list[SearchResult]: ...
    async def delete(self, ids) -> None: ...
```

内置 `InMemoryVectorStore`（纯 Python 余弦相似度）和 `ChromaVectorStore`（Chroma 后端）。

### 3.5 Planner System（计划系统）

#### Plan DAG

```python
@dataclass
class PlanStep:
    id: str
    description: str
    dependencies: list[str]  # 前置步骤 ID
    status: "pending" | "running" | "done" | "failed" | "skipped"
    result: str | None

@dataclass
class Plan:
    goal: str
    steps: list[PlanStep]

    def ready_steps(self) -> list[PlanStep]:
        # 返回所有依赖已完成的 pending 步骤
```

`ready_steps()` 实现了 DAG 拓扑调度：找出所有 status="pending" 且 dependencies 全部 status="done" 的步骤。

#### Planner Protocol

```python
class Planner(Protocol):
    async def plan(self, goal: str, context: PlanContext) -> Plan: ...
    async def replan(self, plan: Plan, feedback: str, context: PlanContext) -> Plan: ...
```

| 实现 | 行为 |
|---|---|
| `LLMPlanner` | LLM 生成 JSON 计划，replan 时保留已完成步骤 |
| `StaticPlanner` | 用户预定义步骤，线性依赖链 |

---

## 4. Orchestrator Layer（编排层）

在 Agent 之上组合多个 Agent 协同工作。

### Pipeline

```
Agent A → Agent B → Agent C
```

线性链，前一个 Agent 的输出作为后一个的输入。可选 `transform` 回调在传递间修改数据。

### Graph

```
     ┌─ Agent B ─┐
A ──>│           │──> D
     └─ Agent C ─┘
```

DAG 并行编排：
1. 计算每轮 ready 节点（依赖全部完成）
2. `asyncio.create_task` 并行执行
3. 收集结果，循环直到全部完成
4. 支持 Edge 上的 `transform` 函数
5. 循环依赖检测（返回 "Graph stuck" 错误）

### Supervisor

协调者-工人模式：
1. 自动注入 `delegate_task` 工具到 coordinator
2. Coordinator 通过 tool_call 将子任务分配给 worker
3. Worker 独立执行并返回结果

---

## 5. Eval Layer（评估层）

### 评估流程

```
EvalCase[] → EvalRunner → Agent.clone().run() → Evaluator.evaluate() → EvalReport
```

`EvalRunner` 使用 `asyncio.Semaphore` 控制并发度，每个 case 使用 `agent.clone()` 确保独立。

### 评估器

| 评估器 | 评判逻辑 |
|---|---|
| `ExactMatch` | 去除空白后完全匹配 |
| `Contains` | 输出包含期望字符串（支持大小写选项） |
| `LLMJudge` | LLM 打分 0-1，超过阈值为通过 |
| `ToolUseEvaluator` | 检查是否使用了指定的工具 |

---

## 关键设计决策

### 为什么用 Protocol 而不是 ABC？

Protocol 实现结构化子类型（鸭子类型的静态版本），实现类不需要继承或注册，只要方法签名匹配即可。这让第三方扩展无需依赖 lattice 包。

### 为什么 dataclass + Pydantic 混用？

- `@dataclass`：所有内部数据类型（Message, Content, Span, Plan 等），轻量无运行时开销
- `Pydantic BaseModel`：仅用于工具参数（需要 `model_json_schema()` 生成 JSON Schema 给 LLM）

### 为什么 Agent.run() 是同步循环而非递归？

`for step_count in range(max_steps)` 的平坦循环比递归更容易理解和调试，也天然提供了最大步数保护。

### 为什么 DeepSeek 继承 OpenAI Provider？

DeepSeek API 完全兼容 OpenAI 格式，只需修改 `base_url` 和 `api_key`，复用全部消息转换和流式解析逻辑。

### 为什么 Content 用 Union 而非继承？

```python
Content = Union[TextContent, ImageContent, ThinkingContent, ToolCall, ToolResult]
```

显式 Union + `isinstance` 检查比继承层级更直观，mypy 能穷举检查分支覆盖，JSON 序列化也更简单。

---

## 数据流图

### 单 Agent 请求流

```
User Input
  │
  ▼
Agent.run(input)
  │
  ├── Memory.retrieve(input) → memory_context
  │
  ├── 构建 system_prompt + memory_context
  │
  └── Loop (max_steps):
        │
        Strategy.step(AgentContext)
          │
          ├── LLMProvider.stream(model, messages, tools)
          │     │
          │     ├── StreamStart
          │     ├── TextDelta / ToolCallStart / ToolCallDelta / ToolCallEnd
          │     └── StreamEnd(ModelResponse)
          │
          ├── 解析 ToolCall → Tool.execute() → ToolResult
          │
          └── StepResult(messages, action=Continue|Finish)
        │
        if Finish: break
  │
  ├── Memory.store(summary)
  │
  └── AgentResult(output, messages, steps, usage)
```

### Graph 编排流

```
Graph.run(input)
  │
  └── while pending:
        │
        ├── ready = nodes where all deps done
        │
        ├── asyncio.gather(
        │     Agent_A.run(input),
        │     Agent_B.run(input),
        │   )
        │
        └── collect outputs → update done set
  │
  └── GraphResult(outputs, final_output)
```
