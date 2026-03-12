"""WorkflowRunner — executes multi-step workflows."""

from __future__ import annotations

import asyncio
import inspect
import time
from typing import TYPE_CHECKING, Optional, cast

from .types import InferenceStep, ToolRoundStep, TransformStep, Workflow, WorkflowResult

if TYPE_CHECKING:
    from octomil.responses.responses import OctomilResponses
    from octomil.responses.tools.executor import ToolExecutor


class WorkflowRunner:
    def __init__(
        self,
        responses: OctomilResponses,
        executor: Optional[ToolExecutor] = None,
    ) -> None:
        self._responses = responses
        self._executor = executor

    async def run(self, workflow: Workflow, input: str) -> WorkflowResult:
        from octomil.responses.types import ResponseRequest, TextOutput, text_input

        start = time.monotonic()
        current_text = input
        outputs = []

        for step in workflow.steps:
            if isinstance(step, InferenceStep):
                request = ResponseRequest(
                    model=step.model,
                    input=[text_input(current_text)],
                    instructions=step.instructions,
                    max_output_tokens=step.max_output_tokens,
                )
                response = await self._responses.create(request)
                outputs.append(response)
                current_text = "".join(item.text for item in response.output if isinstance(item, TextOutput))

            elif isinstance(step, ToolRoundStep):
                if self._executor is None:
                    raise RuntimeError("ToolExecutor required for ToolRoundStep")
                from octomil.responses.tools.runner import ToolRunner

                runner = ToolRunner(self._responses, self._executor, max_iterations=step.max_iterations)
                request = ResponseRequest(
                    model=step.model,
                    input=[text_input(current_text)],
                    tools=step.tools,
                )
                response = await runner.run(request)
                outputs.append(response)
                current_text = "".join(item.text for item in response.output if isinstance(item, TextOutput))

            elif isinstance(step, TransformStep):
                result = step.transform(current_text)
                if inspect.isawaitable(result):
                    current_text = await cast(asyncio.Future[str], result)
                else:
                    current_text = cast(str, result)

        elapsed_ms = (time.monotonic() - start) * 1000
        return WorkflowResult(outputs=outputs, total_latency_ms=elapsed_ms)
