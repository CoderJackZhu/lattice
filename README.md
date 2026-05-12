# Lattice

轻量级、可组合的 Python Agent 框架，专为 AI Agent 算法研究与快速实验设计。

## 特性

- **多模型支持** — 内置 DeepSeek（默认）、OpenAI、Anthropic 接入，统一的 Provider 协议可扩展任意 LLM
- **灵活的 Agent 策略** — ReAct、Plan-and-Execute、Reflexion 三种策略开箱即用，支持自定义 Strategy
- **三层记忆系统** — WorkingMemory（短期 FIFO）、EpisodicMemory（持久化 JSON + 向量检索）、SemanticMemory（纯向量），CompositeMemory 加权融合
- **计划与执行** — Plan DAG 支持依赖拓扑与并行执行，LLMPlanner 自动生成计划，StaticPlanner 手动编排
- **多 Agent 编排** — Pipeline（线性链）、Graph（DAG 并行）、Supervisor（协调者-工人委派）
- **工具与中间件** — `@tool` 装饰器自动生成参数 Schema，ToolKit 管理工具集，Retry / Cache / Timeout 中间件洋葱模型
- **评估框架** — ExactMatch、Contains、LLMJudge、ToolUse 评估器 + EvalRunner 并发批量测试
- **可观测性** — Tracer / Span 追踪，支持 JSON 导出
- **类型安全** — 全局 `mypy --strict`，Protocol 驱动的扩展点

## 快速开始

### 安装

```bash
# 使用 uv（推荐）
uv add lattice

# 从源码安装
git clone <repo-url> && cd lattice
uv sync
```

### 环境变量

```bash
# DeepSeek（默认模型）
export DEEPSEEK_API_KEY="your-key"

# 可选
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
```

### 最简示例

```python
import asyncio
from lattice import Agent

async def main():
    agent = Agent(name="assistant", system_prompt="你是一个有用的助手。")
    result = await agent.run("Python 的 GIL 是什么？")
    print(result.output)

asyncio.run(main())
```

默认使用 `deepseek:deepseek-v4-pro`，也可以指定其他模型：

```python
agent = Agent(name="assistant", model="openai:gpt-4o")
agent = Agent(name="assistant", model="anthropic:claude-sonnet-4-6")
```

### 工具调用

```python
from lattice import Agent, tool
from lattice.tool.builtins.shell import shell
from lattice.tool.builtins.file import read_file, list_dir

agent = Agent(
    name="coder",
    system_prompt="你是一个编程助手。",
    tools=[shell, read_file, list_dir],
)
result = await agent.run("当前目录有哪些 Python 文件？")
```

自定义工具只需一个装饰器：

```python
@tool
async def search_docs(query: str, limit: int = 10) -> str:
    """搜索文档库"""
    # 你的实现 ...
    return f"Found {limit} results for '{query}'"
```

### 记忆系统

```python
from lattice import Agent, CompositeMemory, WorkingMemory, EpisodicMemory

memory = CompositeMemory([
    (WorkingMemory(max_items=20), 1.0),
    (EpisodicMemory(store_path="~/.lattice/memory.json"), 0.7),
])

agent = Agent(name="memo", memory=memory)
result = await agent.run("记住：项目截止日期是下周五")
result = await agent.run("截止日期是什么时候？")  # 能回忆上文
```

### Plan-and-Execute

```python
from lattice import Agent, PlanAndExecuteStrategy, LLMPlanner
from lattice.tool.builtins.shell import shell

planner = LLMPlanner()
agent = Agent(
    name="planner",
    tools=[shell],
    strategy=PlanAndExecuteStrategy(planner=planner),
)
result = await agent.run("创建 hello.py 并运行它")
```

### 多 Agent 编排

```python
from lattice import Agent, Pipeline, Graph, Node, Supervisor

# Pipeline — 线性链
pipeline = Pipeline(agents=[researcher, writer, reviewer])
result = await pipeline.run("分析 Python 3.12 新特性")

# Graph — DAG 并行
graph = Graph(
    nodes=[
        Node(id="research", agent=researcher),
        Node(id="code", agent=coder),
        Node(id="merge", agent=writer, dependencies=["research", "code"]),
    ],
    output_node="merge",
)
result = await graph.run("构建一个 CLI 工具")

# Supervisor — 协调者委派
supervisor = Supervisor(
    coordinator=manager,
    workers={"coder": coder, "reviewer": reviewer},
)
result = await supervisor.run("实现快速排序并审查代码")
```

### 评估

```python
from lattice import Agent, EvalRunner, EvalCase, ExactMatch, Contains

cases = [
    EvalCase(input="1+1=?", expected="2"),
    EvalCase(input="Python 创造者？", expected="Guido"),
]

runner = EvalRunner(agent=agent, evaluator=Contains())
report = await runner.run(cases)
print(report.summary())
# Eval Report: 2/2 passed (100.00% avg score)
```

## 项目结构

```
lattice/
├── agent/              # Agent 核心 + 策略
│   ├── agent.py        #   Agent 类（run / run_stream / clone）
│   ├── strategy.py     #   ReAct / PlanAndExecute / Reflexion
│   └── types.py        #   AgentContext, Action, StepResult, AgentResult
├── llm/                # LLM 抽象层
│   ├── provider.py     #   LLMProvider Protocol + ProviderRegistry
│   ├── providers/      #   DeepSeek / OpenAI / Anthropic 实现
│   ├── types.py        #   Message, Content, StreamEvent 类型
│   └── config.py       #   YAML 配置加载
├── memory/             # 三层记忆
│   ├── working.py      #   WorkingMemory（FIFO + 关键词匹配）
│   ├── episodic.py     #   EpisodicMemory（JSON 持久化 + 可选向量检索）
│   ├── semantic.py     #   SemanticMemory（VectorStore 封装）
│   ├── composite.py    #   CompositeMemory（加权融合）
│   └── stores/         #   InMemoryVectorStore / ChromaVectorStore
├── planner/            # 计划系统
│   ├── base.py         #   Plan / PlanStep DAG + Planner Protocol
│   ├── llm_planner.py  #   LLMPlanner（LLM 生成计划）
│   └── static_planner.py  # StaticPlanner（手动线性计划）
├── orchestrator/       # 多 Agent 编排
│   ├── pipeline.py     #   Pipeline（线性链）
│   ├── graph.py        #   Graph（DAG 并行执行）
│   └── supervisor.py   #   Supervisor（协调者-工人模式）
├── tool/               # 工具系统
│   ├── tool.py         #   @tool 装饰器 + Tool 类
│   ├── toolkit.py      #   ToolKit（工具集 + 中间件管道）
│   ├── middleware.py    #   Retry / Cache / Timeout / Sandbox
│   └── builtins/       #   shell / file / web 内置工具
├── eval/               # 评估框架
│   ├── types.py        #   EvalCase / EvalResult / EvalReport
│   ├── evaluators.py   #   ExactMatch / Contains / LLMJudge / ToolUse
│   └── runner.py       #   EvalRunner（并发批量执行）
└── trace/              # 可观测性
    ├── tracer.py       #   Tracer / Span
    └── exporters.py    #   JSON / OTLP 导出
```

## 配置

可选的 `lattice.yaml` 配置文件：

```yaml
models:
  default: "deepseek:deepseek-v4-pro"
  reasoning: "anthropic:claude-sonnet-4-6"
  fast: "deepseek:deepseek-v4-pro"

providers:
  deepseek:
    api_key: "${DEEPSEEK_API_KEY}"
  openai:
    api_key: "${OPENAI_API_KEY}"
  anthropic:
    api_key: "${ANTHROPIC_API_KEY}"
```

搜索路径：`./lattice.yaml` → `~/.lattice/config.yaml`，支持 `${ENV_VAR:default}` 语法。

## 开发

```bash
# 安装开发依赖
uv sync --group dev

# 运行测试（79 个测试）
uv run pytest tests/ -v

# 类型检查
uv run mypy lattice/

# 代码格式化
uv run ruff check lattice/ --fix
```

## 设计原则

1. **Protocol 驱动** — 所有扩展点（LLMProvider, Memory, Planner, Strategy, VectorStore, Evaluator）均为 Python Protocol，无需继承
2. **可组合** — 每个组件独立使用，自由组合成更复杂的系统
3. **最小依赖** — 核心仅依赖 pydantic / pyyaml / httpx / openai，高级功能按需安装
4. **dataclass 优先** — 数据类型用 `@dataclass`，仅工具参数使用 Pydantic（需要 JSON Schema）
5. **类型安全** — 全量 `mypy --strict`，显式的 Union 类型替代隐式多态

## License

MIT
