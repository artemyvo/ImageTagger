"""Input validation utilities for LLM configuration parameters.

Provides centralized parsing and validation for timeout, retry count, and
query downscaling parameters used in both main window and merge dialog.

Pure Python validation logic (no Qt dependencies).
"""

from __future__ import annotations

from typing import Callable
from imagetagger.providers.llm_provider import LlmProviderError


class InputValidator:
    """Input validation utilities for LLM parameters."""

    @staticmethod
    def parse_timeout_seconds(
        text: str,
        on_error: Callable[[str], None] | None = None,
    ) -> float:
        """Parse and validate timeout input (seconds).
        
        Args:
            text: Raw input text
            on_error: Optional callback for error messages
            
        Returns:
            Timeout as float
            
        Raises:
            LlmProviderError: If input is invalid
        """
        raw_value = text.strip()
        if not raw_value:
            error_msg = "Enter timeout in seconds."
            if on_error:
                on_error(error_msg)
            raise LlmProviderError(error_msg)

        try:
            timeout = int(raw_value)
        except ValueError as exc:
            error_msg = "Timeout must be a whole number of seconds."
            if on_error:
                on_error(error_msg)
            raise LlmProviderError(error_msg) from exc

        if timeout < 1:
            error_msg = "Timeout must be at least 1 second."
            if on_error:
                on_error(error_msg)
            raise LlmProviderError(error_msg)

        return float(timeout)

    @staticmethod
    def parse_retry_count(text: str) -> int:
        """Parse retry count, defaulting to 0 on invalid input.
        
        Args:
            text: Raw input text
            
        Returns:
            Retry count (non-negative)
        """
        raw_value = text.strip()
        try:
            retries = int(raw_value)
        except ValueError:
            return 0
        return max(0, retries)

    @staticmethod
    def parse_max_resolution_mpx(
        text: str,
        on_error: Callable[[str], None] | None = None,
    ) -> float:
        """Parse and validate query downscale (megapixels).
        
        Args:
            text: Raw input text
            on_error: Optional callback for error messages
            
        Returns:
            Max resolution in megapixels
            
        Raises:
            LlmProviderError: If input is invalid
        """
        raw_value = text.strip()
        if not raw_value:
            error_msg = "Enter query downscale in megapixels."
            if on_error:
                on_error(error_msg)
            raise LlmProviderError(error_msg)

        try:
            value = float(raw_value)
        except ValueError as exc:
            error_msg = "Query downscale must be a number."
            if on_error:
                on_error(error_msg)
            raise LlmProviderError(error_msg) from exc

        if value <= 0:
            error_msg = "Query downscale must be greater than 0."
            if on_error:
                on_error(error_msg)
            raise LlmProviderError(error_msg)

        return value

    @staticmethod
    def format_megapixels(value: float) -> str:
        """Format megapixel value for display (3 decimal places).
        
        Args:
            value: Megapixel value to format
            
        Returns:
            Formatted string with trailing zeros stripped
        """
        normalized = f"{float(value):.3f}".rstrip("0").rstrip(".")
        return normalized + ".0" if "." not in normalized else normalized
