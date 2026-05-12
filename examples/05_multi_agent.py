import asyncio

from lattice import Agent
from lattice.orchestrator import Pipeline, Supervisor


async def main():
    researcher = Agent(
        name="researcher",
        model="openai:gpt-4o-mini",
        system_prompt="你是一个研究员，负责收集和总结信息。",
    )

    writer = Agent(
        name="writer",
        model="openai:gpt-4o-mini",
        system_prompt="你是一个技术作家，将研究内容写成清晰的报告。",
    )

    print("=== Pipeline: 研究 → 写作 ===")
    pipeline = Pipeline(agents=[researcher, writer])
    result = await pipeline.run("介绍 Python asyncio 的核心概念")
    print(result.final_output[:500])

    print("\n=== Supervisor: 协调多个 worker ===")
    coder = Agent(
        name="coder",
        model="openai:gpt-4o-mini",
        system_prompt="你是一个 Python 程序员。回复简洁的代码。",
    )

    reviewer = Agent(
        name="reviewer",
        model="openai:gpt-4o-mini",
        system_prompt="你是一个代码审查员。检查代码质量并给出建议。",
    )

    coordinator = Agent(
        name="coordinator",
        model="openai:gpt-4o-mini",
        system_prompt=(
            "你是一个项目协调员。使用 delegate_task 工具将任务分配给 worker。"
            "先让 coder 写代码，再让 reviewer 审查。最后总结结果。"
        ),
    )

    supervisor = Supervisor(
        coordinator=coordinator,
        workers={"coder": coder, "reviewer": reviewer},
    )

    result = await supervisor.run("写一个 Python 快速排序函数，并审查代码质量")
    print(result.output[:500])


if __name__ == "__main__":
    asyncio.run(main())
