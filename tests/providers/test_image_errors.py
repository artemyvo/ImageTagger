"""An image that cannot be prepared fails its own request as a provider error."""
from __future__ import annotations

import pytest

from imagetagger.providers import ollama, openai_compat
from imagetagger.providers.llm_provider import LlmProviderError


@pytest.mark.parametrize(
    "encode",
    [ollama._encode_image, openai_compat._encode_image_data_url],
    ids=["ollama", "openai_compat"],
)
def test_unreadable_image_is_a_provider_error_without_backoff(tmp_path, encode):
    """Batch loops retry or skip on LlmProviderError; anything else would stop the whole batch."""
    with pytest.raises(LlmProviderError) as raised:
        encode(tmp_path / "deleted.png")
    assert "Could not read image file" in str(raised.value)
    assert raised.value.no_backoff
