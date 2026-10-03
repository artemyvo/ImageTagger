from __future__ import annotations

import os
import queue as _queue
import threading as _threading
from pathlib import Path
from typing import Callable, Optional, TypeVar
import tempfile

_T = TypeVar("_T")


def atomic_write_text(path: Path, content: str, encoding: str = "utf-8", durable: bool = True) -> None:
    """Atomically replace a text file via same-directory temp file and os.replace.

    With *durable* False the fsync calls are skipped: the replace is still
    atomic, but a power loss may lose it.  Bulk edits of many files use this;
    fsync makes them orders of magnitude slower.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as handle:
            handle.write(content)
            handle.flush()
            if durable:
                os.fsync(handle.fileno())

        os.replace(tmp_path, path)
        if not durable:
            return

        # Best-effort durability of the directory entry after rename.
        try:
            dir_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Background write queue
#
# Every write to an annotation or sidecar file goes through one worker thread,
# so a later write can never be overtaken by an earlier one that is still
# queued.  ``bg_write_text`` enqueues and returns; ``run_in_write_order``
# enqueues and waits, for callers that need the result (or the error) now.
# ---------------------------------------------------------------------------
_bg_write_queue: _queue.SimpleQueue[Optional[Callable[[], None]]] = _queue.SimpleQueue()
_bg_write_thread: Optional[_threading.Thread] = None
_bg_write_start_lock = _threading.Lock()
_bg_write_error_handler: Optional[Callable[[Path, BaseException], None]] = None


def set_bg_write_error_handler(handler: Callable[[Path, BaseException], None] | None) -> None:
    """Register *handler* to be told about failed background writes.

    It is called on the background thread with the target path and the error.
    """
    global _bg_write_error_handler
    _bg_write_error_handler = handler


def _report_bg_write_error(path: Path, error: BaseException) -> None:
    handler = _bg_write_error_handler
    if handler is None:
        print(f"Background write failed: {path}: {error}", flush=True)
        return
    try:
        handler(path, error)
    except Exception:
        pass


def _bg_write_worker() -> None:
    while True:
        job = _bg_write_queue.get()
        if job is None:
            return
        try:
            job()
        except Exception:
            pass  # jobs report their own failures


def _enqueue_write_job(job: Callable[[], None]) -> None:
    global _bg_write_thread
    if _bg_write_thread is None or not _bg_write_thread.is_alive():
        with _bg_write_start_lock:
            if _bg_write_thread is None or not _bg_write_thread.is_alive():
                t = _threading.Thread(
                    target=_bg_write_worker, daemon=True, name="bg-write"
                )
                t.start()
                _bg_write_thread = t
    _bg_write_queue.put(job)


def bg_write_text(
    path: Path,
    content: str,
    encoding: str = "utf-8",
    on_complete: Callable[[BaseException | None], None] | None = None,
    durable: bool = True,
) -> None:
    """Enqueue a write to be completed on a background thread.

    Uses the same atomic-write logic as ``atomic_write_text`` but does not
    block the calling thread.  Suitable for auto-save operations where the
    in-memory state is already authoritative before the write.

    *on_complete* is called on the background thread after the write, with
    ``None`` on success or the exception on failure.  Failures are also
    passed to the handler set with ``set_bg_write_error_handler``.
    """

    def job() -> None:
        error: BaseException | None = None
        try:
            if durable:
                atomic_write_text(path, content, encoding=encoding)
            else:
                atomic_write_text(path, content, encoding=encoding, durable=False)
        except Exception as exc:
            error = exc
        if on_complete is not None:
            try:
                on_complete(error)
            except Exception:
                pass
        if error is not None:
            _report_bg_write_error(path, error)

    _enqueue_write_job(job)


def run_in_write_order(fn: Callable[[], _T]) -> _T:
    """Run *fn* on the write thread after every write queued before it, and wait.

    Returns *fn*'s result or re-raises its exception in the caller.
    """
    if _threading.current_thread() is _bg_write_thread:
        return fn()

    done = _threading.Event()
    outcome: list = [None, None]

    def job() -> None:
        try:
            outcome[0] = fn()
        except BaseException as exc:
            outcome[1] = exc
        finally:
            done.set()

    _enqueue_write_job(job)
    done.wait()
    if outcome[1] is not None:
        raise outcome[1]
    return outcome[0]


def write_text_in_order(path: Path, content: str, encoding: str = "utf-8") -> None:
    """``atomic_write_text`` that waits for queued background writes to land first."""
    run_in_write_order(lambda: atomic_write_text(path, content, encoding=encoding))
