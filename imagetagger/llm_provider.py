from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Protocol

from imagetagger.llm_queries import LlmQueryError


DEFAULT_LLM_TIMEOUT = 300.0


class LlmProviderError(LlmQueryError):
    pass


class LlmProviderCancelled(LlmProviderError):
    pass


class LlmRequestCancellation:
    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._active_resources: set[object] = set()

    def cancel(self) -> None:
        self._event.set()
        with self._lock:
            resources = list(self._active_resources)
            self._active_resources.clear()
        for resource in resources:
            close = getattr(resource, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled():
            raise LlmProviderCancelled("Request stopped.")

    def set_active_resource(self, resource: object) -> None:
        with self._lock:
            self._active_resources.add(resource)
            already_cancelled = self._event.is_set()
        if already_cancelled:
            close = getattr(resource, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
            raise LlmProviderCancelled("Request stopped.")

    def clear_active_resource(self, resource: object) -> None:
        with self._lock:
            self._active_resources.discard(resource)


class VisionLlmSession(Protocol):
    def generate(
        self,
        image_path: Path,
        prompt: str,
        *,
        timeout: float,
        cancellation: LlmRequestCancellation | None = None,
    ) -> str: ...


class VisionLlmProvider(Protocol):
    display_name: str
    default_endpoint: str

    def normalize_endpoint(self, endpoint: str) -> str: ...

    def fetch_models(self, endpoint: str, *, timeout: float) -> list[str]: ...

    def create_session(self, endpoint: str, model_name: str) -> VisionLlmSession: ...


@dataclass(frozen=True)
class _OllamaSession:
    endpoint: str
    model_name: str

    def generate(
        self,
        image_path: Path,
        prompt: str,
        *,
        timeout: float,
        cancellation: LlmRequestCancellation | None = None,
    ) -> str:
        from imagetagger.ollama import OllamaConnection, generate_with_image

        return generate_with_image(
            OllamaConnection(self.endpoint, self.model_name),
            image_path,
            prompt,
            timeout=timeout,
            cancellation=cancellation,
        )


class _DefaultVisionProvider:
    display_name = "Ollama"

    @property
    def default_endpoint(self) -> str:
        from imagetagger.ollama import DEFAULT_OLLAMA_SERVER

        return DEFAULT_OLLAMA_SERVER

    def normalize_endpoint(self, endpoint: str) -> str:
        from imagetagger.ollama import normalize_server_url

        return normalize_server_url(endpoint)

    def fetch_models(self, endpoint: str, *, timeout: float) -> list[str]:
        from imagetagger.ollama import fetch_models

        return fetch_models(endpoint, timeout=timeout)

    def create_session(self, endpoint: str, model_name: str) -> VisionLlmSession:
        return _OllamaSession(endpoint=endpoint, model_name=model_name)


DEFAULT_VISION_PROVIDER: VisionLlmProvider = _DefaultVisionProvider()