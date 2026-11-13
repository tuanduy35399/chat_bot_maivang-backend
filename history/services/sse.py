from __future__ import annotations

import json
from collections.abc import Mapping
from collections.abc import Callable, Iterable, Iterator
from math import isfinite
from queue import Empty, Queue
from threading import Thread


def encode_sse_event(payload: Mapping[str, object]) -> str:
    """Encode one JSON payload as a complete server-sent event frame."""
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"data: {data}\n\n"


def keep_alive_frame() -> str:
    """Return a comment frame that keeps an otherwise quiet stream open."""
    return ": keep-alive\n\n"


def stream_operation(
    operation: Callable[[], Mapping[str, object]],
    *,
    progress_payloads: Iterable[Mapping[str, object]],
    heartbeat_seconds: float = 10.0,
    error_payload: Callable[[Exception], Mapping[str, object]],
) -> Iterator[str]:
    """Yield progress and terminal SSE frames around a blocking operation."""
    if not isinstance(heartbeat_seconds, (int, float)) or not isfinite(heartbeat_seconds) or heartbeat_seconds <= 0:
        raise ValueError("heartbeat_seconds must be a finite positive number")

    def encode_error(error: Exception) -> str:
        try:
            return encode_sse_event(error_payload(error))
        except Exception:
            return encode_sse_event({
                "type": "error",
                "message": "SSE stream failed.",
            })

    for payload in progress_payloads:
        yield encode_sse_event(payload)

    result_queue: Queue[tuple[str, object]] = Queue(maxsize=1)

    def run_operation() -> None:
        try:
            result_queue.put(("result", operation()))
        except Exception as error:  # Convert worker failures into terminal SSE data.
            result_queue.put(("error", error))

    worker = Thread(target=run_operation, name="chat-sse-operation", daemon=True)
    worker.start()

    while True:
        try:
            kind, value = result_queue.get(timeout=heartbeat_seconds)
        except Empty:
            yield keep_alive_frame()
            continue
        if kind == "error":
            yield encode_error(value)
        else:
            try:
                yield encode_sse_event(value)
            except Exception as error:
                yield encode_error(error)
        return
