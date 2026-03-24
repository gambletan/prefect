import concurrent.futures
import contextvars
import threading
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from prefect._internal.concurrency.api import create_detached_call
from prefect._internal.concurrency.calls import Call
from prefect._internal.concurrency.threads import EventLoopThread, WorkerThread


def identity(x):
    return x


async def aidentity(x):
    return x


def test_event_loop_thread_with_failure_in_start():
    event_loop_thread = EventLoopThread()

    # Simulate a failure during loop thread start
    event_loop_thread._ready_future.set_result = MagicMock(
        side_effect=ValueError("test")
    )

    # The error should propagate to the main thread
    with pytest.raises(ValueError, match="test"):
        event_loop_thread.start()


def test_event_loop_thread_start_race_condition():
    event_loop_thread = EventLoopThread()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for _ in range(10):
            executor.submit(event_loop_thread.start)

    event_loop_thread.shutdown()


def test_event_loop_thread_with_on_shutdown_hook():
    event_loop_thread = EventLoopThread()
    mock = AsyncMock()

    event_loop_thread.start()
    event_loop_thread.add_shutdown_call(Call.new(mock))
    mock.assert_not_called()

    event_loop_thread.shutdown()
    event_loop_thread.thread.join()
    mock.assert_awaited_once()


def test_event_loop_thread_with_on_shutdown_hooks():
    event_loop_thread = EventLoopThread()
    mock = AsyncMock()

    event_loop_thread.start()
    for i in range(5):
        event_loop_thread.add_shutdown_call(Call.new(mock, i))
    mock.assert_not_called()

    event_loop_thread.shutdown()
    event_loop_thread.thread.join()

    mock.assert_has_awaits(call(i) for i in range(5))


def test_event_loop_thread_clears_shutdown_hooks_after_shutdown():
    event_loop_thread = EventLoopThread()

    event_loop_thread.start()
    for i in range(5):
        event_loop_thread.add_shutdown_call(Call.new(AsyncMock(), i))
    event_loop_thread.shutdown()
    event_loop_thread.thread.join()

    assert not event_loop_thread._on_shutdown


@pytest.mark.parametrize("thread_cls", [WorkerThread, EventLoopThread])
@pytest.mark.parametrize("daemon", [True, False])
def test_thread_daemon(daemon, thread_cls):
    thread = thread_cls(daemon=daemon)
    thread.start()
    assert thread.thread.daemon is daemon
    thread.shutdown()


@pytest.mark.parametrize("thread_cls", [WorkerThread, EventLoopThread])
def test_thread_run_once(thread_cls):
    thread = thread_cls(run_once=True)
    thread.start()

    call = Call.new(identity, 1)
    thread.submit(call)

    with pytest.raises(
        RuntimeError,
        match="Worker configured to only run once. A call has already been submitted.",
    ):
        thread.submit(Call.new(identity, 1))

    assert call.future.result() == 1


@pytest.mark.parametrize("thread_cls", [WorkerThread, EventLoopThread])
@pytest.mark.parametrize("work", [identity, aidentity])
def test_thread_submit(work, thread_cls):
    thread = thread_cls()
    call = thread.submit(Call.new(work, 1))
    assert call.result() == 1
    thread.shutdown()


def test_event_loop_thread_submit_preserves_async_call_context():
    marker = contextvars.ContextVar("marker", default=None)
    thread = EventLoopThread()
    thread.start()

    async def read_marker():
        return marker.get()

    token = marker.set("caller")
    try:
        call = thread.submit(Call.new(read_marker))
        assert call.result() == "caller"
    finally:
        marker.reset(token)
        thread.shutdown()
        thread.thread.join()


def test_event_loop_thread_submit_detaches_submitter_context_for_detached_calls():
    thread = EventLoopThread()
    thread.start()

    started = threading.Event()
    submitted = threading.Event()
    release = threading.Event()
    call_ref: dict[str, Call[None]] = {}

    async def mark_started() -> None:
        started.set()

    def submit_while_context_is_entered() -> None:
        context = contextvars.copy_context()

        def inner() -> None:
            call_ref["call"] = thread.submit(create_detached_call(mark_started))
            submitted.set()
            release.wait(timeout=5)

        context.run(inner)

    submitter = threading.Thread(target=submit_while_context_is_entered)
    submitter.start()

    try:
        assert submitted.wait(timeout=2)
        assert started.wait(timeout=2)
        assert call_ref["call"].result(timeout=2) is None
    finally:
        release.set()
        submitter.join()
        thread.shutdown()
        thread.thread.join()
