import asyncio
import contextvars
import uuid

import pytest

import prefect._waiters as waiters_module
from prefect import flow
from prefect._waiters import FlowRunWaiter
from prefect.client.orchestration import PrefectClient
from prefect.flow_engine import run_flow_async
from prefect.server.events.pipeline import EventsPipeline
from prefect.states import Pending

TEST_CONTEXTVAR = contextvars.ContextVar("TEST_CONTEXTVAR")


class TestFlowRunWaiter:
    @pytest.fixture(autouse=True)
    def teardown(self):
        yield

        FlowRunWaiter.instance().stop()

    def test_instance_returns_singleton(self):
        assert FlowRunWaiter.instance() is FlowRunWaiter.instance()

    def test_instance_returns_instance_after_stop(self):
        instance = FlowRunWaiter.instance()
        instance.stop()
        assert FlowRunWaiter.instance() is not instance

    @pytest.mark.timeout(20)
    async def test_wait_for_flow_run(
        self, prefect_client: PrefectClient, emitting_events_pipeline: EventsPipeline
    ):
        """This test will fail with a timeout error if waiting is not working correctly."""

        @flow
        async def test_flow():
            await asyncio.sleep(1)

        flow_run = await prefect_client.create_flow_run(test_flow, state=Pending())
        asyncio.create_task(run_flow_async(flow=test_flow, flow_run=flow_run))

        await FlowRunWaiter.wait_for_flow_run(flow_run.id)

        await emitting_events_pipeline.process_events()

        flow_run = await prefect_client.read_flow_run(flow_run.id)
        assert flow_run.state
        assert flow_run.state.is_completed()

    async def test_wait_for_flow_run_with_timeout(self, prefect_client: PrefectClient):
        @flow
        async def test_flow():
            await asyncio.sleep(5)

        flow_run = await prefect_client.create_flow_run(test_flow, state=Pending())
        run = asyncio.create_task(run_flow_async(flow=test_flow, flow_run=flow_run))

        await FlowRunWaiter.wait_for_flow_run(flow_run.id, timeout=1)

        # FlowRunWaiter stopped waiting before the task finished
        assert not run.done()
        await run

    async def test_wait_for_flow_run_wait_does_not_inherit_creator_context(
        self, monkeypatch
    ):
        missing = object()
        observed_wait_contexts: list[object] = []
        finished_event = asyncio.Event()

        class DummySubscriber:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return None

            def __aiter__(self):
                return self

            async def __anext__(self):
                await asyncio.Event().wait()
                raise StopAsyncIteration

        async def fake_wait_for_call_in_loop_thread(call, *args, **kwargs):
            if call.fn is asyncio.Event:
                return finished_event

            observed_wait_contexts.append(call.context.get(TEST_CONTEXTVAR, missing))
            finished_event.set()
            return await call.fn()

        monkeypatch.setattr(
            waiters_module,
            "get_events_subscriber",
            lambda **_: DummySubscriber(),
        )
        monkeypatch.setattr(
            waiters_module.from_async,
            "wait_for_call_in_loop_thread",
            fake_wait_for_call_in_loop_thread,
        )

        token = TEST_CONTEXTVAR.set("creator-context")
        try:
            await FlowRunWaiter.wait_for_flow_run(uuid.uuid4())
        finally:
            TEST_CONTEXTVAR.reset(token)

        assert observed_wait_contexts == [missing]

    @pytest.mark.timeout(20)
    async def test_non_singleton_mode(
        self, prefect_client: PrefectClient, emitting_events_pipeline: EventsPipeline
    ):
        waiter = FlowRunWaiter()
        assert waiter is not FlowRunWaiter.instance()

        @flow
        async def test_flow():
            await asyncio.sleep(1)

        flow_run = await prefect_client.create_flow_run(test_flow, state=Pending())
        asyncio.create_task(run_flow_async(flow=test_flow, flow_run=flow_run))

        await waiter.wait_for_flow_run(flow_run.id)

        await emitting_events_pipeline.process_events()

        flow_run = await prefect_client.read_flow_run(flow_run.id)
        assert flow_run.state
        assert flow_run.state.is_completed()

        waiter.stop()

    @pytest.mark.timeout(20)
    async def test_handles_concurrent_task_runs(
        self, prefect_client: PrefectClient, emitting_events_pipeline: EventsPipeline
    ):
        @flow
        async def fast_flow():
            await asyncio.sleep(1)

        @flow
        async def slow_flow():
            await asyncio.sleep(5)

        flow_run_1 = await prefect_client.create_flow_run(fast_flow, state=Pending())
        flow_run_2 = await prefect_client.create_flow_run(slow_flow, state=Pending())

        asyncio.create_task(run_flow_async(flow=fast_flow, flow_run=flow_run_1))
        asyncio.create_task(run_flow_async(flow=slow_flow, flow_run=flow_run_2))

        await FlowRunWaiter.wait_for_flow_run(flow_run_1.id)

        await emitting_events_pipeline.process_events()

        flow_run_1 = await prefect_client.read_flow_run(flow_run_1.id)
        flow_run_2 = await prefect_client.read_flow_run(flow_run_2.id)

        assert flow_run_1.state
        assert flow_run_1.state.is_completed()

        assert flow_run_2.state
        assert not flow_run_2.state.is_completed()

        await FlowRunWaiter.wait_for_flow_run(flow_run_2.id)

        await emitting_events_pipeline.process_events()

        flow_run_1 = await prefect_client.read_flow_run(flow_run_1.id)
        flow_run_2 = await prefect_client.read_flow_run(flow_run_2.id)

        assert flow_run_1.state
        assert flow_run_1.state.is_completed()

        assert flow_run_2.state
        assert flow_run_2.state.is_completed()
