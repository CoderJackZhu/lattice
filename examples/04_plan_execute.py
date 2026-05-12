import asyncio

from lattice import Agent
from lattice.agent.strategy import PlanAndExecuteStrategy
from lattice.planner import LLMPlanner
from lattice.tool.builtins.file import list_dir, read_file, write_file
from lattice.tool.builtins.shell import shell


async def main():
    planner = LLMPlanner(model="openai:gpt-4o-mini")

    agent = Agent(
        name="planner_demo",
        model="openai:gpt-4o-mini",
        system_prompt="你是一个擅长规划和执行的编程助手。",
        tools=[shell, read_file, write_file, list_dir],
        strategy=PlanAndExecuteStrategy(planner=planner),
    )

    result = await agent.run(
        "在当前目录创建一个 hello.py 文件，内容是打印 'Hello, Lattice!'，然后运行它验证输出。"
    )
    print(result.output)
    print(f"\n[Steps: {len(result.steps)}]")


if __name__ == "__main__":
    asyncio.run(main())
