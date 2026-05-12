from __future__ import annotations

import json
import re

from lattice.llm.provider import registry
from lattice.llm.types import Message, TextContent
from lattice.planner.base import Plan, PlanContext, PlanStep

DEFAULT_PLAN_PROMPT = """You are a task planner. Given a goal and available tools, create a structured plan.

Available tools: {tools}

Goal: {goal}

Respond with ONLY a JSON array of steps. Each step:
- "id": unique string (e.g., "step_1")
- "description": what to do
- "dependencies": list of step ids that must complete first

Example:
[
  {{"id": "step_1", "description": "Search for info", "dependencies": []}},
  {{"id": "step_2", "description": "Analyze results", "dependencies": ["step_1"]}}
]"""

DEFAULT_REPLAN_PROMPT = """You are a task planner. A plan needs revision based on feedback.

Current plan state:
{plan_state}

Feedback: {feedback}

Create a revised plan. Keep completed steps, modify or replace failed/pending steps.
Respond with ONLY a JSON array of ALL steps (including completed ones with their status)."""


def _extract_json_array(text: str) -> list[dict]:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(text)


class LLMPlanner:
    def __init__(self, model: str = "deepseek:deepseek-v4-pro", plan_prompt: str = DEFAULT_PLAN_PROMPT) -> None:
        self._model = model
        self._plan_prompt = plan_prompt
        self._provider, self._model_name = registry.from_model_id(model)

    async def plan(self, goal: str, context: PlanContext) -> Plan:
        tools_desc = ", ".join(t.name for t in context.available_tools) if context.available_tools else "none"
        prompt = self._plan_prompt.format(tools=tools_desc, goal=goal)

        messages = [Message(role="user", content=[TextContent(text=prompt)])]
        response = await self._provider.complete(self._model_name, messages, temperature=0.0)

        text = "".join(
            c.text for c in response.message.content if isinstance(c, TextContent)
        )

        try:
            steps_data = _extract_json_array(text)
        except (json.JSONDecodeError, ValueError):
            steps_data = [{"id": "step_1", "description": goal, "dependencies": []}]

        steps = [
            PlanStep(
                id=s.get("id", f"step_{i+1}"),
                description=s.get("description", ""),
                dependencies=s.get("dependencies", []),
            )
            for i, s in enumerate(steps_data)
        ]
        return Plan(goal=goal, steps=steps)

    async def replan(self, plan: Plan, feedback: str, context: PlanContext) -> Plan:
        plan_state = json.dumps([
            {"id": s.id, "description": s.description, "status": s.status, "result": s.result}
            for s in plan.steps
        ], indent=2, ensure_ascii=False)

        prompt = DEFAULT_REPLAN_PROMPT.format(plan_state=plan_state, feedback=feedback)
        messages = [Message(role="user", content=[TextContent(text=prompt)])]
        response = await self._provider.complete(self._model_name, messages, temperature=0.0)

        text = "".join(
            c.text for c in response.message.content if isinstance(c, TextContent)
        )

        try:
            steps_data = _extract_json_array(text)
        except (json.JSONDecodeError, ValueError):
            return plan

        done_ids = {s.id for s in plan.steps if s.status == "done"}
        new_steps: list[PlanStep] = []
        for s_data in steps_data:
            sid = s_data.get("id", "")
            if sid in done_ids:
                original = next((s for s in plan.steps if s.id == sid), None)
                if original:
                    new_steps.append(original)
                    continue
            new_steps.append(PlanStep(
                id=sid,
                description=s_data.get("description", ""),
                dependencies=s_data.get("dependencies", []),
                status=s_data.get("status", "pending") if sid in done_ids else "pending",
            ))

        return Plan(goal=plan.goal, steps=new_steps)
