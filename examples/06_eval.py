import asyncio

from lattice import Agent
from lattice.eval import Contains, EvalCase, EvalRunner, ExactMatch


async def main():
    agent = Agent(
        name="math_agent",
        model="deepseek:deepseek-v4-pro",
        system_prompt="你是一个数学助手。只回答数字结果，不要解释。",
    )

    cases = [
        EvalCase(input="1 + 1 = ?", expected="2"),
        EvalCase(input="10 * 5 = ?", expected="50"),
        EvalCase(input="100 / 4 = ?", expected="25"),
        EvalCase(input="Python 的创造者是谁？", expected="Guido van Rossum"),
    ]

    print("=== ExactMatch 评估 ===")
    runner = EvalRunner(agent=agent, evaluator=ExactMatch())
    report = await runner.run(cases[:3])
    print(report.summary())

    print("\n=== Contains 评估 ===")
    runner2 = EvalRunner(agent=agent, evaluator=Contains())
    report2 = await runner2.run(cases[3:])
    print(report2.summary())


if __name__ == "__main__":
    asyncio.run(main())
