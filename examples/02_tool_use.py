import asyncio

from lattice import Agent
from lattice.tool.builtins.shell import shell
from lattice.tool.builtins.file import list_dir, read_file


async def main():
    agent = Agent(
        name="assistant",
        model="openai:gpt-4o-mini",
        system_prompt="你是一个有用的编程助手。",
        tools=[shell, read_file, list_dir],
    )
    result = await agent.run("当前目录下有哪些文件？列出来并统计数量。")
    print(result.output)
    print(f"\n[Steps: {len(result.steps)}, Tokens: {result.usage.input_tokens + result.usage.output_tokens}]")


if __name__ == "__main__":
    asyncio.run(main())
