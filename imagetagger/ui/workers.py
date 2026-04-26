from __future__ import annotations

from typing import Callable
from PyQt6.QtCore import QObject, pyqtSignal
from imagetagger.providers.llm_provider import LlmProviderCancelled, LlmProviderError

class RegenerateWorker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, task: Callable[[Callable[[str], None]], object]) -> None:
        super().__init__()
        self.task = task

    def run(self) -> None:
        try:
            result = self.task(self.progress.emit)
        except LlmProviderCancelled as exc:
            self.cancelled.emit(str(exc))
            return
        except LlmProviderError as exc:
            self.failed.emit(str(exc))
            return
        except Exception as exc:
            self.failed.emit(f"Unexpected LLM error: {exc}")
            return
        self.finished.emit(result)
