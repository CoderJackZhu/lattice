from __future__ import annotations

from lattice.planner.base import Plan, PlanContext, PlanStep


class StaticPlanner:
    def __init__(self, steps: list[str]) -> None:
        self._steps = steps

    async def plan(self, goal: str, context: PlanContext) -> Plan:
        plan_steps = [
            PlanStep(
                id=f"step_{i+1}",
                description=desc,
                dependencies=[f"step_{i}"] if i > 0 else [],
            )
            for i, desc in enumerate(self._steps)
        ]
        return Plan(goal=goal, steps=plan_steps)

    async def replan(self, plan: Plan, feedback: str, context: PlanContext) -> Plan:
        return plan
