from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Protocol
from urllib.parse import urlparse

from imagetagger.utils.llm_queries import LlmQueryError


DEFAULT_LLM_TIMEOUT = 300.0


class LlmProviderError(LlmQueryError):
    # When True, auto-threading should not treat this error as a performance
    # signal (i.e. no backoff / thread-count reduction on retry).
    no_backoff: bool = False


class LlmProviderCancelled(LlmProviderError):
    pass


def normalize_server_url(
    server: str,
    default_server: str,
    allowed_schemes: set[str] | None = None,
) -> str:
    """Normalize and validate server URL for LLM provider.
    
    Args:
        server: Raw server URL/host input
        default_server: Default server URL if input is empty
        allowed_schemes: Set of allowed URL schemes (default: http, https)
        
    Returns:
        Normalized server URL with scheme and without trailing slash
        
    Raises:
        LlmProviderError: If URL is invalid
    """
    if allowed_schemes is None:
        allowed_schemes = {"http", "https"}
    
    trimmed = server.strip()
    if not trimmed:
        return default_server

    if "://" not in trimmed:
        trimmed = f"http://{trimmed}"

    parsed = urlparse(trimmed)
    if parsed.scheme not in allowed_schemes or not parsed.netloc:
        raise LlmProviderError("Enter a valid server address.")

    normalized = f"{parsed.scheme}://{parsed.netloc}"
    return normalized.rstrip("/")


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
        thread_count: int | None = None,
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
        thread_count: int | None = None,
    ) -> str:
        from imagetagger.providers.ollama import OllamaConnection, generate_with_image

        return generate_with_image(
            OllamaConnection(self.endpoint, self.model_name),
            image_path,
            prompt,
            timeout=timeout,
            cancellation=cancellation,
            thread_count=thread_count,
        )


class _DefaultVisionProvider:
    display_name = "LLM"

    @property
    def default_endpoint(self) -> str:
        from imagetagger.providers.ollama import DEFAULT_OLLAMA_SERVER

        return DEFAULT_OLLAMA_SERVER

    def normalize_endpoint(self, endpoint: str) -> str:
        from imagetagger.providers.ollama import DEFAULT_OLLAMA_SERVER
        from imagetagger.providers.openai_compat import DEFAULT_OPENAI_COMPAT_SERVER

        # Try to determine if this looks like an Ollama endpoint by checking the port
        # or use OpenAI-compat defaults as a fallback
        try:
            parsed = urlparse(endpoint.strip() or DEFAULT_OLLAMA_SERVER)
            is_ollama_port = parsed.port == 11434
        except ValueError:
            is_ollama_port = False

        # Use appropriate defaults based on the apparent server type
        default_server = DEFAULT_OLLAMA_SERVER if is_ollama_port else DEFAULT_OPENAI_COMPAT_SERVER

        try:
            return normalize_server_url(endpoint, default_server)
        except LlmProviderError:
            raise
        except Exception as exc:
            raise LlmProviderError(str(exc)) from exc

    @staticmethod
    def _uses_ollama_interface(endpoint: str) -> bool:
        parsed = urlparse(endpoint)
        try:
            return parsed.port == 11434
        except ValueError:
            return False

    def fetch_models(self, endpoint: str, *, timeout: float) -> list[str]:
        normalized = self.normalize_endpoint(endpoint)
        if self._uses_ollama_interface(normalized):
            from imagetagger.providers.ollama import fetch_models

            return fetch_models(normalized, timeout=timeout)

        from imagetagger.providers.openai_compat import fetch_models

        return fetch_models(normalized, timeout=timeout)

    def create_session(self, endpoint: str, model_name: str) -> VisionLlmSession:
        normalized = self.normalize_endpoint(endpoint)
        if self._uses_ollama_interface(normalized):
            return _OllamaSession(endpoint=normalized, model_name=model_name)

        return _OpenAiCompatSession(endpoint=normalized, model_name=model_name)


@dataclass(frozen=True)
class _OpenAiCompatSession:
    endpoint: str
    model_name: str

    def generate(
        self,
        image_path: Path,
        prompt: str,
        *,
        timeout: float,
        cancellation: LlmRequestCancellation | None = None,
        thread_count: int | None = None,
    ) -> str:
        from imagetagger.providers.openai_compat import OpenAiCompatConnection, generate_with_image

        return generate_with_image(
            OpenAiCompatConnection(self.endpoint, self.model_name),
            image_path,
            prompt,
            timeout=timeout,
            cancellation=cancellation,
        )


DEFAULT_VISION_PROVIDER: VisionLlmProvider = _DefaultVisionProvider()