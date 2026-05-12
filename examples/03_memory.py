import asyncio

from lattice import Agent
from lattice.memory import CompositeMemory, EpisodicMemory, WorkingMemory
from lattice.tool.builtins.shell import shell


async def main():
    memory = CompositeMemory([
        (WorkingMemory(max_items=20), 1.0),
        (EpisodicMemory(store_path="~/.lattice/demo_memory.json"), 0.7),
    ])

    agent = Agent(
        name="memory_demo",
        model="deepseek:deepseek-v4-pro",
        system_prompt="你是一个有记忆能力的助手。",
        tools=[shell],
        memory=memory,
    )

    result = await agent.run("当前目录有哪些文件？记住这些信息。")
    print("=== 第一次 ===")
    print(result.output)

    result = await agent.run("之前我问过你什么？你还记得吗？")
    print("\n=== 第二次 ===")
    print(result.output)


if __name__ == "__main__":
    asyncio.run(main())
