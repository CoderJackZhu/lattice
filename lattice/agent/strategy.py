from __future__ import annotations

import json
from typing import Protocol

from pydantic import ValidationError

from lattice.agent.types import (
    AgentContext,
    Continue,
    Finish,
    ReflectionJudge,
    StepResult,
)
from lattice.llm.types import (
    Message,
    StreamEnd,
    TextContent,
    ToolCall,
    ToolResult,
)
from lattice.planner.base import Plan, PlanContext, Planner
from lattice.tool.tool import ToolContext


class Strategy(Protocol):
    async def step(self, ctx: AgentContext) -> StepResult: ...


async def _execute_tool_calls(
    tool_calls: list[ToolCall], ctx: AgentContext
) -> list[ToolResult]:
    tools_by_name = {t.name: t for t in ctx.tools} if ctx.tools else {}
    results: list[ToolResult] = []

    for tc in tool_calls:
        tool_obj = tools_by_name.get(tc.name)
        if not tool_obj:
            results.append(ToolResult(
                tool_call_id=tc.id,
                content=f"Error: tool '{tc.name}' not found",
                is_error=True,
            ))
            continue

        try:
            validated = tool_obj.parameters(**tc.arguments)
        except (ValidationError, TypeError) as e:
            results.append(ToolResult(
                tool_call_id=tc.id,
                content=f"Error validating parameters: {e}",
                is_error=True,
            ))
            continue

        try:
            tool_ctx = ToolContext(tool_call_id=tc.id)
            output = await tool_obj.execute(validated, tool_ctx)
            content = output.content if isinstance(output.content, str) else json.dumps(output.content)
            results.append(ToolResult(tool_call_id=tc.id, content=content))
        except Exception as e:
            results.append(ToolResult(
                tool_call_id=tc.id,
                content=f"Error executing tool: {e}",
                is_error=True,
            ))

    return results


async def _call_llm_and_parse(ctx: AgentContext) -> tuple[Message | None, list[ToolCall]]:
    assert ctx.stream_fn is not None
    response = None
    async for event in ctx.stream_fn(
        ctx.model,
        ctx.messages,
        system=ctx.system_prompt or None,
        tools=[t.to_schema() for t in ctx.tools] if ctx.tools else None,
    ):
        if isinstance(event, StreamEnd):
            response = event.response

    if response is None:
        return None, []

    assistant_msg = response.message
    tool_calls = [c for c in assistant_msg.content if isinstance(c, ToolCall)]
    return assistant_msg, tool_calls


class ReActStrategy:
    async def step(self, ctx: AgentContext) -> StepResult:
        assistant_msg, tool_calls = await _call_llm_and_parse(ctx)

        if assistant_msg is None:
            return StepResult(messages=[], action=Finish(output="Error: no response from LLM"))

        if not tool_calls:
            text = "".join(
                c.text for c in assistant_msg.content if isinstance(c, TextContent)
            )
            return StepResult(messages=[assistant_msg], action=Finish(output=text))

        tool_results = await _execute_tool_calls(tool_calls, ctx)
        tool_msg = Message(role="tool", content=tool_results)  # type: ignore[arg-type]
        return StepResult(messages=[assistant_msg, tool_msg], action=Continue())


class PlanAndExecuteStrategy:
    def __init__(self, planner: Planner) -> None:
        self._planner = planner
        self._plan: Plan | None = None

    async def step(self, ctx: AgentContext) -> StepResult:
        if self._plan is None:
            goal = self._extract_goal(ctx.messages)
            plan_ctx = PlanContext(
                available_tools=[t.to_schema() for t in ctx.tools] if ctx.tools else [],
                memory_context=ctx.memory_context,
                messages=ctx.messages,
            )
            self._plan = await self._planner.plan(goal, plan_ctx)
            plan_text = self._format_plan(self._plan)
            msg = Message(role="assistant", content=[TextContent(text=plan_text)])
            return StepResult(messages=[msg], action=Continue())

        ready = self._plan.ready_steps()

        if not ready:
            all_done = all(s.status in ("done", "skipped") for s in self._plan.steps)
            if all_done:
                output = self._summarize_results(self._plan)
                msg = Message(role="assistant", content=[TextContent(text=output)])
                return StepResult(messages=[msg], action=Finish(output=output))
            else:
                failed = [s for s in self._plan.steps if s.status == "failed"]
                if failed:
                    feedback = "\n".join(f"Step {s.id} failed: {s.result}" for s in failed)
                    plan_ctx = PlanContext(
                        available_tools=[t.to_schema() for t in ctx.tools] if ctx.tools else [],
                        memory_context=ctx.memory_context,
                        messages=ctx.messages,
                    )
                    self._plan = await self._planner.replan(self._plan, feedback, plan_ctx)
                    return StepResult(messages=[], action=Continue())
                output = "Plan stuck: no ready steps and not all done"
                return StepResult(messages=[], action=Finish(output=output))

        current_step = ready[0]
        current_step.status = "running"

        step_prompt = f"Execute this step: {current_step.description}"
        step_messages = list(ctx.messages) + [
            Message(role="user", content=[TextContent(text=step_prompt)])
        ]

        step_ctx = AgentContext(
            messages=step_messages,
            tools=ctx.tools,
            model=ctx.model,
            system_prompt=ctx.system_prompt,
            memory_context=ctx.memory_context,
            step_count=ctx.step_count,
            max_steps=ctx.max_steps,
            stream_fn=ctx.stream_fn,
        )

        assistant_msg, tool_calls = await _call_llm_and_parse(step_ctx)

        new_messages: list[Message] = []
        if assistant_msg is None:
            current_step.status = "failed"
            current_step.result = "No response from LLM"
            return StepResult(messages=[], action=Continue())

        new_messages.append(assistant_msg)

        if tool_calls:
            tool_results = await _execute_tool_calls(tool_calls, step_ctx)
            tool_msg = Message(role="tool", content=tool_results)  # type: ignore[arg-type]
            new_messages.append(tool_msg)
            result_text = "\n".join(
                r.content if isinstance(r.content, str) else str(r.content)
                for r in tool_results
            )
        else:
            result_text = "".join(
                c.text for c in assistant_msg.content if isinstance(c, TextContent)
            )

        current_step.status = "done"
        current_step.result = result_text[:500]

        return StepResult(messages=new_messages, action=Continue())

    def _extract_goal(self, messages: list[Message]) -> str:
        for msg in messages:
            if msg.role == "user":
                return "".join(
                    c.text for c in msg.content if isinstance(c, TextContent)
                )
        return ""

    def _format_plan(self, plan: Plan) -> str:
        lines = [f"Plan for: {plan.goal}", ""]
        for s in plan.steps:
            deps = f" (depends on: {', '.join(s.dependencies)})" if s.dependencies else ""
            lines.append(f"  [{s.id}] {s.description}{deps}")
        return "\n".join(lines)

    def _summarize_results(self, plan: Plan) -> str:
        lines = [f"Completed plan: {plan.goal}", ""]
        for s in plan.steps:
            result_preview = (s.result or "")[:200]
            lines.append(f"  [{s.id}] {s.description} -> {result_preview}")
        return "\n".join(lines)


class ReflexionStrategy:
    def __init__(
        self, judge: ReflectionJudge | None = None, max_reflections: int = 3
    ) -> None:
        self._judge = judge
        self._max_reflections = max_reflections
        self._react = ReActStrategy()
        self._reflection_count = 0
        self._goal: str | None = None

    async def step(self, ctx: AgentContext) -> StepResult:
        result = await self._react.step(ctx)

        if isinstance(result.action, Finish) and self._judge:
            if self._goal is None:
                self._goal = self._extract_goal(ctx.messages)

            verdict = await self._judge.judge(
                output=result.action.output,
                goal=self._goal,
                context=ctx.messages,
            )

            if not verdict.passed and self._reflection_count < self._max_reflections:
                self._reflection_count += 1
                reflection_msg = Message(
                    role="user",
                    content=[TextContent(
                        text=f"Your previous answer was not satisfactory. "
                             f"Feedback: {verdict.feedback}\nPlease try again."
                    )],
                )
                return StepResult(
                    messages=[*result.messages, reflection_msg],
                    action=Continue(),
                )

        return result

    def _extract_goal(self, messages: list[Message]) -> str:
        for msg in messages:
            if msg.role == "user":
                return "".join(
                    c.text for c in msg.content if isinstance(c, TextContent)
                )
        return ""
