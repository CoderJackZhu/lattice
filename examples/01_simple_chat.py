import asyncio

from lattice import Agent


async def main():
    agent = Agent(
        name="chat",
        model="openai:gpt-4o-mini",
        system_prompt="你是一个有用的助手。用中文回答。",
    )
    result = await agent.run("Python 的 GIL 是什么？一句话解释。")
    print(result.output)


if __name__ == "__main__":
    asyncio.run(main())
