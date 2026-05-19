from __future__ import annotations

import os
import queue as _queue
import threading as _threading
from pathlib import Path
from typing import Callable, Optional, Tuple
import tempfile


def atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Atomically replace a text file via same-directory temp file and os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(tmp_path, path)

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
# ---------------------------------------------------------------------------
_BgItem = Tuple[Path, str, str, Optional[Callable[[], None]]]

_bg_write_queue: _queue.SimpleQueue[Optional[_BgItem]] = _queue.SimpleQueue()
_bg_write_thread: Optional[_threading.Thread] = None
_bg_write_start_lock = _threading.Lock()


def _bg_write_worker() -> None:
    while True:
        item = _bg_write_queue.get()
        if item is None:
            return
        path, content, encoding, on_complete = item
        try:
            atomic_write_text(path, content, encoding=encoding)
        except Exception:
            pass  # best-effort; caller already updated in-memory state
        if on_complete is not None:
            try:
                on_complete()
            except Exception:
                pass


def bg_write_text(
    path: Path,
    content: str,
    encoding: str = "utf-8",
    on_complete: Callable[[], None] | None = None,
) -> None:
    """Enqueue a write to be completed on a background thread (best-effort).

    Uses the same atomic-write logic as ``atomic_write_text`` but does not
    block the calling thread.  Suitable for auto-save operations where the
    in-memory state is already authoritative before the write.

    *on_complete* is called on the background thread after the write finishes.
    """
    global _bg_write_thread
    if _bg_write_thread is None or not _bg_write_thread.is_alive():
        with _bg_write_start_lock:
            if _bg_write_thread is None or not _bg_write_thread.is_alive():
                t = _threading.Thread(
                    target=_bg_write_worker, daemon=True, name="bg-write"
                )
                t.start()
                _bg_write_thread = t
    _bg_write_queue.put((path, content, encoding, on_complete))