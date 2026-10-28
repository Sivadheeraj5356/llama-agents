# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from __future__ import annotations

import asyncio
from typing import Any

from workflows import Context, Workflow, step
from workflows.events import (
    Event,
    HumanResponseEvent,
    InputRequiredEvent,
    StartEvent,
    StopEvent,
    WorkflowIdleEvent,
)
from workflows.plugins.basic import BasicRuntime, JournalRecord


class Answer(HumanResponseEvent):
    name: str
    value: int


class GoA(Event):
    pass


class GoB(Event):
    pass


class Got(Event):
    name: str
    value: int


class TwoWaiters(Workflow):
    """Two steps wait for the same event type, each with its own requirements."""

    @step
    async def start(self, ctx: Context, ev: StartEvent) -> GoA | GoB:
        ctx.send_event(GoA())
        ctx.send_event(GoB())

    @step
    async def a(self, ctx: Context, ev: GoA) -> Got:
        answer = await ctx.wait_for_event(
            Answer,
            waiter_event=InputRequiredEvent(),
            waiter_id="a",
            requirements={"name": "a"},
            timeout=None,
        )
        return Got(name="a", value=answer.value)

    @step
    async def b(self, ctx: Context, ev: GoB) -> Got:
        answer = await ctx.wait_for_event(
            Answer,
            waiter_event=InputRequiredEvent(),
            waiter_id="b",
            requirements={"name": "b"},
            timeout=None,
        )
        return Got(name="b", value=answer.value)

    @step
    async def join(self, ctx: Context, ev: Got) -> StopEvent | None:
        events = ctx.collect_events(ev, [Got, Got])
        if events is None:
            return None
        return StopEvent(
            result=sorted((item.name, item.value) for item in events)  # type: ignore[attr-defined]
        )


async def _journal(runtime: BasicRuntime, run_id: str) -> list[JournalRecord]:
    return [record async for record in runtime.journal(run_id)]


async def test_restore_from_journal_keeps_waiter_requirements() -> None:
    runtime = BasicRuntime()
    handler = TwoWaiters(timeout=None, runtime=runtime).run()
    await asyncio.sleep(0)
    records_task = asyncio.create_task(_journal(runtime, handler.run_id))
    async for event in handler.stream_events(expose_internal=True):
        if isinstance(event, WorkflowIdleEvent):
            break
    await handler.cancel_run()
    records = await asyncio.wait_for(records_task, timeout=5)

    workflow = TwoWaiters(timeout=None, runtime=runtime)
    restored = runtime.restore(workflow, None, records)
    handler = workflow.run(ctx=restored)
    await handler.send_event(Answer(name="b", value=2))
    await handler.send_event(Answer(name="a", value=1))

    result: Any = await handler
    # each waiter must receive only the answer that matches its requirements
    assert result == [("a", 1), ("b", 2)]
