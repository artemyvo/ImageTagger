from __future__ import annotations

import base64
import http.client
import json
import math
import threading
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse


DEFAULT_OLLAMA_SERVER = "http://127.0.0.1:11434"
# Vision-capable LLMs can take a while on busy GPUs; keep this generous.
DEFAULT_TIMEOUT = 300.0
# Default to 1 MP; can be overridden from config.json.
MAX_IMAGE_PIXELS_FOR_OLLAMA = 1_000_000

_resize_warning_lock = threading.Lock()
_resize_warning_pending = False


class OllamaError(Exception):
    pass


class OllamaCancelled(OllamaError):
    pass


class OllamaCancellation:
    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._active_connection: http.client.HTTPConnection | None = None

    def cancel(self) -> None:
        self._event.set()
        with self._lock:
            connection = self._active_connection
            self._active_connection = None
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled():
            raise OllamaCancelled("Request stopped.")

    def set_active_connection(self, connection: http.client.HTTPConnection) -> None:
        with self._lock:
            self._active_connection = connection
            already_cancelled = self._event.is_set()
        if already_cancelled:
            try:
                connection.close()
            except Exception:
                pass
            raise OllamaCancelled("Request stopped.")

    def clear_active_connection(self, connection: http.client.HTTPConnection) -> None:
        with self._lock:
            if self._active_connection is connection:
                self._active_connection = None


@dataclass(frozen=True)
class OllamaConnection:
    server_url: str
    model_name: str


def configure_runtime(*, max_image_pixels: int | None = None) -> None:
    global MAX_IMAGE_PIXELS_FOR_OLLAMA

    if max_image_pixels is not None:
        MAX_IMAGE_PIXELS_FOR_OLLAMA = max(1, int(max_image_pixels))


def normalize_server_url(server: str) -> str:
    trimmed = server.strip()
    if not trimmed:
        return DEFAULT_OLLAMA_SERVER

    if "://" not in trimmed:
        trimmed = f"http://{trimmed}"

    parsed = urlparse(trimmed)
    if not parsed.scheme or not parsed.netloc:
        raise OllamaError("Enter a valid Ollama server address.")

    normalized = f"{parsed.scheme}://{parsed.netloc}"
    return normalized.rstrip("/")


def _request_json(
    server: str,
    path: str,
    payload: dict | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    cancellation: OllamaCancellation | None = None,
) -> dict:
    server_url = normalize_server_url(server)
    parsed = urlparse(server_url)
    if parsed.scheme == "http":
        connection_class = http.client.HTTPConnection
    elif parsed.scheme == "https":
        connection_class = http.client.HTTPSConnection
    else:
        raise OllamaError("Only http and https Ollama server URLs are supported.")

    request_path = path if path.startswith("/") else f"/{path}"
    data = None
    headers: dict[str, str] = {}

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    connection = connection_class(parsed.netloc, timeout=timeout)
    response: http.client.HTTPResponse | None = None

    try:
        if cancellation is not None:
            cancellation.raise_if_cancelled()
            cancellation.set_active_connection(connection)

        method = "POST" if payload is not None else "GET"
        connection.request(method, request_path, body=data, headers=headers)
        response = connection.getresponse()

        chunks = bytearray()
        while True:
            if cancellation is not None:
                cancellation.raise_if_cancelled()

            chunk = response.read(64 * 1024)
            if not chunk:
                break
            chunks.extend(chunk)

        response_text = chunks.decode("utf-8")
        if response.status >= 400:
            message = f"Ollama server returned HTTP {response.status}."
            details = response_text.strip()
            if details:
                message = f"{message} {details}"
            raise OllamaError(message)

        return json.loads(response_text)
    except OllamaCancelled:
        raise
    except TimeoutError as exc:
        raise OllamaError(
            f"Timed out after {int(timeout)} seconds while contacting Ollama server. "
            "Try again or use a smaller/faster model."
        ) from exc
    except OSError as exc:
        if cancellation is not None and cancellation.is_cancelled():
            raise OllamaCancelled("Request stopped.") from exc
        raise OllamaError(f"Could not reach Ollama server: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise OllamaError("Ollama server returned invalid JSON.") from exc
    finally:
        if cancellation is not None:
            cancellation.clear_active_connection(connection)
        if response is not None:
            try:
                response.close()
            except Exception:
                pass
        connection.close()


def consume_resize_warning() -> str | None:
    global _resize_warning_pending
    with _resize_warning_lock:
        if not _resize_warning_pending:
            try:
                import PIL  # noqa: F401
            except ImportError:
                _resize_warning_pending = True

        if not _resize_warning_pending:
            return None
        _resize_warning_pending = False
    return (
        "Pillow is not installed, so images are sent to Ollama at original size.\n\n"
        "Install dependencies to enable resizing before upload."
    )


def fetch_models(server: str, timeout: float = 5.0) -> list[str]:
    payload = _request_json(server, "/api/tags", timeout=timeout)

    models = payload.get("models", [])
    if not isinstance(models, list):
        raise OllamaError("Ollama server response did not include a models list.")

    model_names: list[str] = []
    for model in models:
        if isinstance(model, dict):
            name = model.get("name")
            if isinstance(name, str) and name.strip():
                model_names.append(name.strip())

    return model_names


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _load_prompt(filename: str, default: str) -> str:
    """Return prompt text from *filename* in the prompts directory, or *default*."""
    try:
        return (_PROMPTS_DIR / filename).read_text(encoding="utf-8").strip()
    except OSError:
        return default


_DEFAULT_TAGS_PROMPT = (
    "Analyze the image and return a list of short descriptive tags, one tag per line. "
    "Each tag must be one or two words maximum. Use lowercase only. "
    "Do not use commas. "
    "Do not include numbering, explanations, or sentences."
)

_DEFAULT_DESCRIPTION_PROMPT = (
    "Analyze the image and write a single descriptive sentence that captures what is depicted. "
    "Describe the content and key visual elements in detail. "
    "Return only the description sentence, nothing else. "
    "Do not use commas."
)

_DEFAULT_VALIDATION_PROMPT = (
    "You are validating image annotations. The annotation list may include short tags and one long description.\n"
    "The original storage format may use commas only as separators between annotations. Those separator commas are not mistakes.\n"
    "Missing commas are also not a problem.\n"
    "Current annotations, one per line:\n{tags}\n"
    "Analyze the image and verify whether the annotations are accurate and complete.\n"
    "If everything is correct, reply with exactly: OK\n"
    "If there are problems, reply using exactly this plain-text format:\n"
    "ISSUES:\n"
    "<brief explanation of what is wrong>\n"
    "TAGS:\n"
    "<corrected tags, one per line, no commas>\n"
    "DESCRIPTION:\n"
    "<corrected description without commas, or leave blank if no description is needed>\n"
    "Do not return JSON or any extra headings."
)

_PROMPT_DEFAULTS: dict[str, str] = {
    "tagging": _DEFAULT_TAGS_PROMPT,
    "description": _DEFAULT_DESCRIPTION_PROMPT,
    "validation": _DEFAULT_VALIDATION_PROMPT,
}

_PROMPT_FILENAMES: dict[str, str] = {
    "tagging": "tags_prompt.txt",
    "description": "description_prompt.txt",
    "validation": "validation_prompt.txt",
}

# In-memory overrides set via UI "Apply".
_PROMPT_OVERRIDES: dict[str, str] = {}


def _assert_prompt_kind(kind: str) -> None:
    if kind not in _PROMPT_DEFAULTS:
        raise OllamaError(f"Unknown prompt kind: {kind}")


def get_default_prompt(kind: str) -> str:
    _assert_prompt_kind(kind)
    return _PROMPT_DEFAULTS[kind]


def load_prompt_for_kind(kind: str) -> str:
    _assert_prompt_kind(kind)
    return _load_prompt(_PROMPT_FILENAMES[kind], _PROMPT_DEFAULTS[kind])


def prompt_source_for_kind(kind: str) -> str:
    _assert_prompt_kind(kind)
    if kind in _PROMPT_OVERRIDES:
        return "memory"

    prompt_path = _PROMPTS_DIR / _PROMPT_FILENAMES[kind]
    try:
        prompt_path.read_text(encoding="utf-8")
    except OSError:
        return "default"
    return "file"


def set_prompt_override(kind: str, prompt: str) -> None:
    _assert_prompt_kind(kind)
    _PROMPT_OVERRIDES[kind] = prompt.strip()


def clear_prompt_override(kind: str) -> None:
    _assert_prompt_kind(kind)
    _PROMPT_OVERRIDES.pop(kind, None)


def save_prompt_for_kind(kind: str, prompt: str) -> str:
    _assert_prompt_kind(kind)
    text = prompt.strip()
    try:
        _PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
        (_PROMPTS_DIR / _PROMPT_FILENAMES[kind]).write_text(text, encoding="utf-8")
    except OSError as exc:
        raise OllamaError(f"Could not save prompt file: {exc}") from exc
    return text


def reset_prompt_to_default(kind: str) -> str:
    _assert_prompt_kind(kind)
    default_text = _PROMPT_DEFAULTS[kind]
    try:
        _PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
        (_PROMPTS_DIR / _PROMPT_FILENAMES[kind]).write_text(default_text, encoding="utf-8")
    except OSError as exc:
        raise OllamaError(f"Could not reset prompt file: {exc}") from exc
    clear_prompt_override(kind)
    return default_text


def _active_prompt(kind: str) -> str:
    _assert_prompt_kind(kind)
    override = _PROMPT_OVERRIDES.get(kind)
    if override is not None:
        return override
    return load_prompt_for_kind(kind)


def active_prompt_for_kind(kind: str) -> str:
    _assert_prompt_kind(kind)
    return _active_prompt(kind)


def _format_annotations_for_validation(annotations: str) -> str:
    lines = [part.strip() for part in annotations.replace("\n", ",").split(",") if part.strip()]
    return "\n".join(f"- {line}" for line in lines)


def _resize_image_bytes_for_ollama(image_bytes: bytes, max_pixels: int = MAX_IMAGE_PIXELS_FOR_OLLAMA) -> bytes:
    global _resize_warning_pending
    try:
        from PIL import Image, ImageOps, UnidentifiedImageError
    except ImportError:
        # Keep original image if Pillow is not installed yet and emit a one-time warning.
        with _resize_warning_lock:
            _resize_warning_pending = True
        return image_bytes

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image = ImageOps.exif_transpose(image)
            width, height = image.size
            if width <= 0 or height <= 0:
                return image_bytes

            pixels = width * height
            if pixels <= max_pixels:
                return image_bytes

            scale = math.sqrt(max_pixels / float(pixels))
            target_size = (max(1, int(width * scale)), max(1, int(height * scale)))
            resized = image.resize(target_size, Image.Resampling.LANCZOS)

            output = BytesIO()
            image_format = (image.format or "").upper()
            if image_format in {"JPEG", "JPG"}:
                if resized.mode not in {"RGB", "L"}:
                    resized = resized.convert("RGB")
                resized.save(output, format="JPEG", quality=90, optimize=True)
            elif image_format == "PNG":
                resized.save(output, format="PNG", optimize=True)
            else:
                # For uncommon formats, encode as PNG to preserve transparency where possible.
                resized.save(output, format="PNG", optimize=True)

            return output.getvalue()
    except (OSError, ValueError, UnidentifiedImageError):
        return image_bytes


def _encode_image(image_path: Path) -> str:
    try:
        image_bytes = image_path.read_bytes()
    except OSError as exc:
        raise OllamaError(f"Could not read image file: {exc}") from exc
    image_bytes = _resize_image_bytes_for_ollama(image_bytes)
    return base64.b64encode(image_bytes).decode("ascii")


def _generate_with_image(
    server: str,
    model: str,
    image_path: Path,
    prompt: str,
    timeout: float = DEFAULT_TIMEOUT,
    cancellation: OllamaCancellation | None = None,
) -> str:
    payload: dict[str, object] = {
        "model": model,
        "prompt": prompt,
        "images": [_encode_image(image_path)],
        "stream": False,
    }

    payload = _request_json(
        server,
        "/api/generate",
        payload=payload,
        timeout=timeout,
        cancellation=cancellation,
    )
    response = payload.get("response")
    if not isinstance(response, str) or not response.strip():
        raise OllamaError("Ollama returned an empty response.")
    return response.strip()


def generate_tags(
    server: str,
    model: str,
    image_path: Path,
    timeout: float = DEFAULT_TIMEOUT,
    cancellation: OllamaCancellation | None = None,
) -> str:
    prompt = _active_prompt("tagging")
    return _generate_with_image(server, model, image_path, prompt, timeout=timeout, cancellation=cancellation)


def generate_description(
    server: str,
    model: str,
    image_path: Path,
    timeout: float = DEFAULT_TIMEOUT,
    cancellation: OllamaCancellation | None = None,
) -> str:
    prompt = _active_prompt("description")
    return _generate_with_image(server, model, image_path, prompt, timeout=timeout, cancellation=cancellation)


def validate_tags(
    server: str,
    model: str,
    image_path: Path,
    tags: str,
    timeout: float = DEFAULT_TIMEOUT,
    cancellation: OllamaCancellation | None = None,
) -> str:
    prompt_template = _active_prompt("validation")
    prompt = prompt_template.replace("{tags}", _format_annotations_for_validation(tags))
    return _generate_with_image(server, model, image_path, prompt, timeout=timeout, cancellation=cancellation)
