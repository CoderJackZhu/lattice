# Lattice API 参考

## 目录

- [Agent](#agent)
- [Strategy](#strategy)
- [LLM](#llm)
- [Tool](#tool)
- [Memory](#memory)
- [Planner](#planner)
- [Orchestrator](#orchestrator)
- [Eval](#eval)
- [Trace](#trace)
- [Types](#types)

---

## Agent

### `class Agent`

**模块**: `lattice.agent.agent`

核心 Agent 类，封装 LLM 交互循环。

```python
Agent(
    name: str,
    *,
    model: str = "deepseek:deepseek-v4-pro",
    system_prompt: str | Callable[[], str] = "",
    tools: list[Tool] | None = None,
    strategy: Strategy | None = None,
    memory: Memory | None = None,
    max_steps: int = 50,
    max_tokens_per_step: int = 4096,
)
```

**参数**：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `name` | `str` | 必填 | Agent 名称，用于日志和追踪 |
| `model` | `str` | `"deepseek:deepseek-v4-pro"` | 模型 ID，格式 `"provider:model"` |
| `system_prompt` | `str \| Callable` | `""` | 系统提示词，支持字符串或无参函数（动态生成） |
| `tools` | `list[Tool]` | `None` | 可用工具列表 |
| `strategy` | `Strategy` | `ReActStrategy()` | 执行策略 |
| `memory` | `Memory` | `None` | 记忆模块 |
| `max_steps` | `int` | `50` | 最大执行步数 |
| `max_tokens_per_step` | `int` | `4096` | 每步最大 token 数 |

**方法**：

#### `async start(input: str | Message) -> None`

初始化 Agent 会话。清空消息历史，从记忆中检索上下文，追加用户消息。

#### `async step() -> StepResult`

执行一步。必须先调用 `start()`。将当前状态打包为 `AgentContext` 委托给 Strategy。

#### `async run(input: str | Message) -> AgentResult`

完整执行循环。等价于 `start()` + 循环 `step()` 直到 `Finish` 或达到 `max_steps`。完成后自动存储摘要到记忆。

**返回**: `AgentResult`

#### `async run_stream(input: str | Message) -> AsyncIterator[AgentEvent]`

流式执行，逐步 yield `AgentEvent`（agent_start, step_start, step_end, agent_end）。

#### `clone() -> Agent`

浅拷贝 Agent，独立的消息历史和记忆上下文，共享配置和工具。用于并发执行（如评估）。

---

## Strategy

### `class Strategy` (Protocol)

**模块**: `lattice.agent.strategy`

```python
class Strategy(Protocol):
    async def step(self, ctx: AgentContext) -> StepResult: ...
```

### `class ReActStrategy`

默认策略。调用 LLM，有 ToolCall 则执行工具返回 Continue，否则返回 Finish。

```python
ReActStrategy()
```

### `class PlanAndExecuteStrategy`

```python
PlanAndExecuteStrategy(planner: Planner)
```

| 参数 | 类型 | 说明 |
|---|---|---|
| `planner` | `Planner` | 计划生成器（LLMPlanner / StaticPlanner） |

首次 step 生成计划，后续按 DAG 拓扑顺序逐步执行。步骤失败时触发 replan。

### `class ReflexionStrategy`

```python
ReflexionStrategy(
    judge: ReflectionJudge | None = None,
    max_reflections: int = 3,
)
```

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `judge` | `ReflectionJudge` | `None` | 评判协议，评估输出质量 |
| `max_reflections` | `int` | `3` | 最大反思次数 |

在 ReAct 基础上增加自我反思。Finish 时调用 judge，未通过则注入反馈继续。

---

## LLM

### `class LLMProvider` (Protocol)

**模块**: `lattice.llm.provider`

```python
class LLMProvider(Protocol):
    async def stream(
        self, model: str, messages: list[Message], *,
        system: str | None = None,
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        stop: list[str] | None = None,
        response_format: type[BaseModel] | None = None,
    ) -> AsyncIterator[StreamEvent]: ...

    async def complete(self, model: str, messages: list[Message], **kwargs) -> ModelResponse: ...
```

### `class ProviderRegistry`

```python
registry = ProviderRegistry()  # 全局单例
```

**方法**：

| 方法 | 说明 |
|---|---|
| `register(name, factory)` | 注册 Provider 工厂函数 |
| `get(name, **kwargs)` | 获取 Provider 实例（带缓存） |
| `from_model_id(model_id) -> (provider, model_name)` | 解析 `"provider:model"` 格式 |

### 内置 Providers

#### `class DeepSeekProvider`

```python
DeepSeekProvider(api_key: str | None = None, base_url: str | None = None)
```

默认 `base_url="https://api.deepseek.com/v1"`，API key 从 `DEEPSEEK_API_KEY` 环境变量读取。继承 OpenAIProvider。

#### `class OpenAIProvider`

```python
OpenAIProvider(api_key: str | None = None, base_url: str | None = None)
```

完整实现 stream/complete，支持 tool_calls 增量拼接、流式 usage 统计。

#### `class AnthropicProvider`

```python
AnthropicProvider(api_key: str | None = None)
```

需安装 `anthropic` 包：`uv add "lattice[anthropic]"`

### 配置

#### `load_config(path: str | None = None) -> LatticeConfig`

加载 YAML 配置，搜索路径：`path` → `./lattice.yaml` → `~/.lattice/config.yaml`。

```python
@dataclass
class LatticeConfig:
    models: dict[str, str]          # {"default": "deepseek:deepseek-v4-pro", ...}
    providers: dict[str, ProviderConfig]  # {"deepseek": ProviderConfig(...), ...}
```

---

## Tool

### `@tool` 装饰器

**模块**: `lattice.tool.tool`

```python
@tool
async def my_tool(query: str, limit: int = 10) -> str:
    """工具描述"""
    return "result"

# 或者指定描述
@tool(description="自定义描述")
async def another_tool(x: int) -> str:
    return str(x)
```

自动从函数签名生成 Pydantic 模型和 JSON Schema。支持同步和异步函数。

返回类型可以是 `str`（自动包装为 `ToolOutput`）或直接返回 `ToolOutput`。

### `class Tool`

```python
Tool(name: str, description: str, parameters: type[BaseModel], execute_fn: Callable)
```

**方法**：

| 方法 | 说明 |
|---|---|
| `async execute(params, ctx) -> ToolOutput` | 执行工具 |
| `to_schema() -> ToolSchema` | 生成 JSON Schema（发送给 LLM） |

### `class ToolKit`

```python
ToolKit(name: str, tools: list[Tool], middleware: list[ToolMiddleware] | None = None)
```

**方法**：

| 方法 | 说明 |
|---|---|
| `async execute(tool_name, params, ctx) -> ToolOutput` | 通过中间件管道执行工具 |
| `get_tools() -> list[Tool]` | 获取所有工具 |
| `get_schemas() -> list[ToolSchema]` | 获取所有工具 Schema |

### 中间件

**模块**: `lattice.tool.middleware`

```python
class ToolMiddleware(Protocol):
    async def __call__(self, ctx: ToolContext, params: BaseModel, next: ToolExecutor) -> ToolOutput: ...
```

| 类 | 参数 | 说明 |
|---|---|---|
| `RetryMiddleware` | `max_retries=3, backoff=1.0` | 指数退避重试 |
| `CacheMiddleware` | `ttl=300` | TTL 缓存 |
| `TimeoutMiddleware` | `timeout=30.0` | 超时控制 |
| `SandboxMiddleware` | `sandbox_type="subprocess"` | 沙箱执行（预留） |

### 内置工具

| 工具 | 模块 | 参数 |
|---|---|---|
| `shell` | `lattice.tool.builtins.shell` | `command: str, timeout: int = 30` |
| `read_file` | `lattice.tool.builtins.file` | `path: str` |
| `write_file` | `lattice.tool.builtins.file` | `path: str, content: str` |
| `list_dir` | `lattice.tool.builtins.file` | `path: str = "."` |
| `http_get` | `lattice.tool.builtins.web` | `url: str, headers: str = ""` |
| `http_post` | `lattice.tool.builtins.web` | `url: str, body: str = "", content_type: str = "application/json"` |

### 数据类型

```python
@dataclass
class ToolOutput:
    content: str | list[Any] = ""
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class ToolContext:
    agent_name: str = ""
    tool_call_id: str = ""
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    trace_span: Span | None = None
```

---

## Memory

### `class Memory` (Protocol)

**模块**: `lattice.memory.base`

```python
class Memory(Protocol):
    async def retrieve(self, query: str, top_k: int = 5) -> list[MemoryItem]: ...
    async def store(self, item: MemoryItem) -> None: ...
    async def clear(self) -> None: ...
```

### `class MemoryItem`

```python
@dataclass
class MemoryItem:
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    timestamp: float = 0.0
    source: str = ""    # "working" / "episodic" / "semantic"
```

### `class WorkingMemory`

```python
WorkingMemory(max_items: int = 100)
```

内存 FIFO 队列，超过容量自动淘汰最旧条目。检索使用关键词匹配，空查询返回最新的 top_k 条。

### `class EpisodicMemory`

```python
EpisodicMemory(store_path: str, embedding_fn: EmbeddingFn | None = None)
```

| 参数 | 说明 |
|---|---|
| `store_path` | JSON 持久化文件路径 |
| `embedding_fn` | 可选的嵌入函数，提供后启用向量检索 |

无 embedding_fn 时退化为关键词匹配。数据自动持久化到 JSON 文件。

### `class SemanticMemory`

```python
SemanticMemory(backend: VectorStore)
```

VectorStore 的薄封装，将 SearchResult 转换为 MemoryItem。

### `class CompositeMemory`

```python
CompositeMemory(memories: list[tuple[Memory, float]])
```

加权融合多个 Memory 实例。`retrieve()` 并发查询所有子记忆，按 `score * weight` 排序。`store()` 只写入第一个（primary）。`clear()` 清空所有。

### VectorStore Protocol

```python
class VectorStore(Protocol):
    async def add(self, texts: list[str], metadatas: list[dict]) -> list[str]: ...
    async def search(self, query: str, top_k: int = 5) -> list[SearchResult]: ...
    async def delete(self, ids: list[str]) -> None: ...
```

**实现**：

| 类 | 说明 |
|---|---|
| `InMemoryVectorStore(embedding_fn)` | 纯 Python 余弦相似度 |
| `ChromaVectorStore(collection_name, persist_dir)` | Chroma 后端（需 `uv add "lattice[chroma]"`） |

### 类型别名

```python
EmbeddingFn = Callable[[list[str]], Awaitable[list[list[float]]]]
```

---

## Planner

### `class Planner` (Protocol)

**模块**: `lattice.planner.base`

```python
class Planner(Protocol):
    async def plan(self, goal: str, context: PlanContext) -> Plan: ...
    async def replan(self, plan: Plan, feedback: str, context: PlanContext) -> Plan: ...
```

### 数据类型

```python
@dataclass
class PlanStep:
    id: str = ""
    description: str = ""
    dependencies: list[str] = field(default_factory=list)
    status: Literal["pending", "running", "done", "failed", "skipped"] = "pending"
    result: str | None = None

@dataclass
class Plan:
    goal: str = ""
    steps: list[PlanStep] = field(default_factory=list)

    def ready_steps(self) -> list[PlanStep]:
        """返回所有依赖已完成的 pending 步骤"""

@dataclass
class PlanContext:
    available_tools: list[ToolSchema] = field(default_factory=list)
    memory_context: list[MemoryItem] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)
```

### `class LLMPlanner`

```python
LLMPlanner(model: str = "deepseek:deepseek-v4-pro", plan_prompt: str = DEFAULT_PLAN_PROMPT)
```

使用 LLM 生成 JSON 格式的步骤列表。replan 时保留已完成步骤。

### `class StaticPlanner`

```python
StaticPlanner(steps: list[str])
```

预定义步骤列表，自动生成线性依赖链。replan 返回原计划不变。

---

## Orchestrator

### `class Pipeline`

**模块**: `lattice.orchestrator.pipeline`

```python
Pipeline(agents: list[Agent], *, transform: Callable | None = None)
```

| 参数 | 说明 |
|---|---|
| `agents` | 顺序执行的 Agent 列表 |
| `transform` | 可选的数据转换函数 `(current_input: str, previous_outputs: list[str]) -> str` |

#### `async run(input: str) -> PipelineResult`

```python
@dataclass
class PipelineResult:
    outputs: list[str]    # 每个 Agent 的输出
    final_output: str     # 最后一个 Agent 的输出
```

### `class Graph`

**模块**: `lattice.orchestrator.graph`

```python
Graph(
    nodes: list[Node],
    edges: list[Edge] | None = None,
    *,
    entry_node: str | None = None,
    output_node: str | None = None,
)
```

```python
@dataclass
class Node:
    id: str
    agent: Agent
    dependencies: list[str] = field(default_factory=list)

@dataclass
class Edge:
    source: str
    target: str
    transform: Callable[[str], str] | None = None
```

两种定义依赖关系的方式：
1. 通过 `Node.dependencies` 直接指定
2. 通过 `Edge` 列表，支持边上的 `transform`

#### `async run(input: str) -> GraphResult`

```python
@dataclass
class GraphResult:
    outputs: dict[str, str]  # {node_id: output}
    final_output: str        # output_node 的输出
```

### `class Supervisor`

**模块**: `lattice.orchestrator.supervisor`

```python
Supervisor(
    coordinator: Agent,
    workers: dict[str, Agent],
    *,
    max_rounds: int = 10,
)
```

| 参数 | 说明 |
|---|---|
| `coordinator` | 协调者 Agent，自动注入 `delegate_task` 工具 |
| `workers` | 工人 Agent 字典 `{name: agent}` |
| `max_rounds` | 最大交互轮数 |

`delegate_task(worker: str, task: str) -> str` — 协调者通过此工具将子任务分配给指定 worker。

#### `async run(input: str) -> AgentResult`

---

## Eval

### `class EvalRunner`

**模块**: `lattice.eval.runner`

```python
EvalRunner(agent: Agent, evaluator: Evaluator, *, concurrency: int = 5)
```

| 参数 | 说明 |
|---|---|
| `agent` | 被评估的 Agent（每个 case 使用 clone） |
| `evaluator` | 评估器 |
| `concurrency` | 最大并发数 |

#### `async run(cases: list[EvalCase]) -> EvalReport`

### 数据类型

```python
@dataclass
class EvalCase:
    input: str              # 输入文本
    expected: str = ""      # 期望输出
    metadata: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

@dataclass
class EvalResult:
    case: EvalCase
    output: str = ""
    score: float = 0.0      # 0.0 - 1.0
    passed: bool = False
    details: str = ""
    latency_ms: float = 0.0
    steps: int = 0

@dataclass
class EvalReport:
    results: list[EvalResult]
    total: int
    passed: int
    failed: int
    avg_score: float
    avg_latency_ms: float

    def summary(self) -> str:
        """返回人类可读的报告摘要"""
```

### `class Evaluator` (Protocol)

```python
class Evaluator(Protocol):
    async def evaluate(self, output: str, expected: str, **kwargs) -> EvalResult: ...
```

`kwargs` 中可以接收 `case: EvalCase` 和 `messages: list[Message]`。

### 内置评估器

| 类 | 构造参数 | 评判逻辑 |
|---|---|---|
| `ExactMatch` | 无 | `output.strip() == expected.strip()` |
| `Contains` | `case_sensitive: bool = False` | `expected in output`（可选大小写敏感） |
| `LLMJudge` | `model: str, threshold: float = 0.7` | LLM 打分 0-1，超过阈值通过 |
| `ToolUseEvaluator` | `required_tools: list[str]` | 检查消息中是否包含指定工具调用 |

---

## Trace

### `class Tracer`

**模块**: `lattice.trace.tracer`

```python
tracer = Tracer()  # 全局单例
```

**方法**：

| 方法 | 说明 |
|---|---|
| `start_span(name, attributes) -> Span` | 创建新 Span |
| `end_span(span)` | 结束 Span |
| `async with span(name, **attrs) -> Span` | 上下文管理器，自动关联父子 |
| `export(format="json") -> str` | 导出所有 Span |
| `clear()` | 清空所有 Span |

### `class Span`

```python
@dataclass
class Span:
    trace_id: str
    span_id: str
    parent_id: str | None
    name: str
    start_time: float
    end_time: float | None
    attributes: dict[str, Any]
    events: list[SpanEvent]
    status: Literal["ok", "error"]
```

---

## Types

### Content 类型

**模块**: `lattice.llm.types`

```python
Content = TextContent | ImageContent | ThinkingContent | ToolCall | ToolResult
```

| 类型 | 关键字段 | `type` 标识 |
|---|---|---|
| `TextContent` | `text: str` | `"text"` |
| `ImageContent` | `url: str, media_type: str` | `"image"` |
| `ThinkingContent` | `text: str` | `"thinking"` |
| `ToolCall` | `id, name, arguments: dict` | `"tool_call"` |
| `ToolResult` | `tool_call_id, content, is_error` | `"tool_result"` |

### Message

```python
@dataclass
class Message:
    role: Literal["user", "assistant", "tool"]
    content: list[Content]
```

### ModelResponse

```python
@dataclass
class ModelResponse:
    message: Message
    usage: Usage
    stop_reason: str    # "end_turn" / "tool_use" / "max_tokens"
    model: str
    latency_ms: float
```

### Usage

```python
@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
```

### StreamEvent 类型

```python
StreamEvent = StreamStart | TextDelta | ThinkingDelta
            | ToolCallStart | ToolCallDelta | ToolCallEnd
            | StreamEnd | StreamError
```

| 事件 | 关键字段 |
|---|---|
| `StreamStart` | `model` |
| `TextDelta` | `text` |
| `ThinkingDelta` | `text` |
| `ToolCallStart` | `tool_call_id, name` |
| `ToolCallDelta` | `tool_call_id, arguments_fragment` |
| `ToolCallEnd` | `tool_call_id, name, arguments` |
| `StreamEnd` | `response: ModelResponse` |
| `StreamError` | `error: str` |

### Agent 类型

**模块**: `lattice.agent.types`

```python
Action = Continue | Finish

@dataclass
class AgentContext:
    messages: list[Message]
    tools: list[Tool]
    model: str
    system_prompt: str
    memory_context: list[Any]
    step_count: int
    max_steps: int
    stream_fn: Callable | None

@dataclass
class StepResult:
    messages: list[Message]     # 本步新增的消息
    action: Action              # Continue 或 Finish(output=...)

@dataclass
class AgentResult:
    output: str
    messages: list[Message]
    steps: list[StepResult]
    usage: Usage
    trace_id: str

@dataclass
class AgentEvent:
    type: str   # "agent_start"/"agent_end"/"step_start"/"step_end"/...
    data: dict
    timestamp: float

class ReflectionJudge(Protocol):
    async def judge(self, output: str, goal: str, context: list[Message]) -> ReflectionVerdict: ...

@dataclass
class ReflectionVerdict:
    passed: bool = False
    feedback: str = ""
    score: float = 0.0
```
