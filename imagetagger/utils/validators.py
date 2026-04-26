"""Qt-specific validator factories for LLM configuration input fields.

Wraps PyQt6 validators with consistent parameter ranges and formatting.
These factories are used in UI setup to attach validators to QLineEdit widgets.
"""

from __future__ import annotations

from PyQt6.QtGui import QDoubleValidator, QIntValidator
from PyQt6.QtWidgets import QWidget


def create_timeout_validator(parent: QWidget) -> QIntValidator:
    """Create a QIntValidator for timeout input (1 to 86400 seconds).
    
    Args:
        parent: Parent Qt widget
        
    Returns:
        QIntValidator configured for timeout seconds
    """
    return QIntValidator(1, 86400, parent)


def create_retry_validator(parent: QWidget) -> QIntValidator:
    """Create a QIntValidator for retry count (0 to 10).
    
    Args:
        parent: Parent Qt widget
        
    Returns:
        QIntValidator configured for retry count
    """
    return QIntValidator(0, 10, parent)


def create_max_resolution_validator(parent: QWidget) -> QDoubleValidator:
    """Create a QDoubleValidator for query downscale in megapixels.
    
    Args:
        parent: Parent Qt widget
        
    Returns:
        QDoubleValidator configured for megapixels (0.01 to 1000.0)
    """
    validator = QDoubleValidator(0.01, 1000.0, 3, parent)
    validator.setNotation(QDoubleValidator.Notation.StandardNotation)
    return validator


def create_threads_validator(parent: QWidget) -> QIntValidator:
    """Create a QIntValidator for thread count (0 to 128).
    
    Args:
        parent: Parent Qt widget
        
    Returns:
        QIntValidator configured for thread count
    """
    return QIntValidator(0, 128, parent)
