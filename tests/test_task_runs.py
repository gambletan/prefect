import asyncio
import contextvars
import threading
import uuid

import pytest

import prefect.task_runs as task_runs_module
from prefect import task
from prefect.task_engine import run_task_async
from prefect.task_runs import TaskRunWaiter

TEST_CONTEXTVAR = contextvars.ContextVar("TEST_CONTEXTVAR", default=None)


class TestTaskRunWaiter:
    @pytest.fixture(autouse=True)
    def teardown(self):
        yield

        TaskRunWaiter.instance().stop()

    def test_instance_returns_singleton(self):
        assert TaskRunWaiter.instance() is TaskRunWaiter.instance()

    def test_instance_returns_instance_after_stop(self):
        instance = TaskRunWaiter.instance()
        instance.stop()
        assert TaskRunWaiter.instance() is not instance

    def test_consumer_task_does_not_inherit_creator_context(self, monkeypatch):
        started = threading.Event()
        observed: dict[str, str | None] = {}

        class DummySubscriber:
            async def __aenter__(self):
                observed["value"] = TEST_CONTEXTVAR.get()
                started.set()
                return self

            async def __aexit__(self, *_):
                return None

            def __aiter__(self):
                return self

            async def __anext__(self):
                await asyncio.Event().wait()
                raise StopAsyncIteration

        monkeypatch.setattr(
            task_runs_module,
            "get_events_subscriber",
            lambda **_: DummySubscriber(),
        )

        token = TEST_CONTEXTVAR.set("creator-context")
        try:
            TaskRunWaiter.instance()
            assert started.wait(timeout=5)
        finally:
            TEST_CONTEXTVAR.reset(token)

        assert observed["value"] is None

    @pytest.mark.timeout(20)
    @pytest.mark.usefixtures("use_hosted_api_server")
    async def test_wait_for_task_run(self, prefect_client, emitting_events_pipeline):
        """This test will fail with a timeout error if waiting is not working correctly."""

        @task
        async def test_task():
            await asyncio.sleep(1)

        task_run_id = uuid.uuid4()
        asyncio.create_task(run_task_async(task=test_task, task_run_id=task_run_id))

        await TaskRunWaiter.wait_for_task_run(task_run_id)

        await emitting_events_pipeline.process_events()

        task_run = await prefect_client.read_task_run(task_run_id)
        assert task_run.state.is_completed()

    @pytest.mark.usefixtures("use_hosted_api_server")
    async def test_wait_for_task_run_with_timeout(self):
        @task
        async def test_task():
            await asyncio.sleep(5)

        task_run_id = uuid.uuid4()
        run = asyncio.create_task(
            run_task_async(task=test_task, task_run_id=task_run_id)
        )

        await TaskRunWaiter.wait_for_task_run(task_run_id, timeout=1)

        # TaskRunWaiter stopped waiting before the task finished
        assert not run.done()
        await run

    async def test_wait_for_task_run_wait_does_not_inherit_creator_context(
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
            task_runs_module,
            "get_events_subscriber",
            lambda **_: DummySubscriber(),
        )
        monkeypatch.setattr(
            task_runs_module.from_async,
            "wait_for_call_in_loop_thread",
            fake_wait_for_call_in_loop_thread,
        )

        token = TEST_CONTEXTVAR.set("creator-context")
        try:
            await TaskRunWaiter.wait_for_task_run(uuid.uuid4())
        finally:
            TEST_CONTEXTVAR.reset(token)

        assert observed_wait_contexts == [missing]

    @pytest.mark.timeout(20)
    @pytest.mark.usefixtures("use_hosted_api_server")
    async def test_non_singleton_mode(self, prefect_client, emitting_events_pipeline):
        waiter = TaskRunWaiter()
        assert waiter is not TaskRunWaiter.instance()

        @task
        async def test_task():
            await asyncio.sleep(1)

        task_run_id = uuid.uuid4()
        asyncio.create_task(run_task_async(task=test_task, task_run_id=task_run_id))

        await waiter.wait_for_task_run(task_run_id)

        await emitting_events_pipeline.process_events()

        task_run = await prefect_client.read_task_run(task_run_id)
        assert task_run.state.is_completed()

        waiter.stop()

    @pytest.mark.timeout(20)
    @pytest.mark.usefixtures("use_hosted_api_server")
    async def test_handles_concurrent_task_runs(
        self, prefect_client, emitting_events_pipeline
    ):
        @task
        async def fast_task():
            await asyncio.sleep(1)

        @task
        async def slow_task():
            await asyncio.sleep(5)

        task_run_id_1 = uuid.uuid4()
        task_run_id_2 = uuid.uuid4()

        asyncio.create_task(run_task_async(task=fast_task, task_run_id=task_run_id_1))
        asyncio.create_task(run_task_async(task=slow_task, task_run_id=task_run_id_2))

        await TaskRunWaiter.wait_for_task_run(task_run_id_1)

        await emitting_events_pipeline.process_events()

        task_run_1 = await prefect_client.read_task_run(task_run_id_1)
        task_run_2 = await prefect_client.read_task_run(task_run_id_2)

        assert task_run_1.state.is_completed()
        assert not task_run_2.state.is_completed()

        await TaskRunWaiter.wait_for_task_run(task_run_id_2)

        await emitting_events_pipeline.process_events()

        task_run_1 = await prefect_client.read_task_run(task_run_id_1)
        task_run_2 = await prefect_client.read_task_run(task_run_id_2)

        assert task_run_1.state.is_completed()
        assert task_run_2.state.is_completed()
