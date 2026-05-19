from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Sentinel meaning "the validated timestamp has not been loaded yet".
# Distinct from None, which means "loaded and no timestamp present".
_UNKNOWN = object()

@dataclass
class ImageRecord:
    image_path: Path
    text_path: Path
    text: str
    _sidecar_has_pending_fixup: bool | None = field(default=None, repr=False)
    _resolution_mpx: float | None = field(default=None, repr=False)
    # _UNKNOWN means "not yet loaded"; None means "loaded, no validated timestamp"
    _sidecar_validated: object = field(default_factory=lambda: _UNKNOWN, repr=False)

    @property
    def has_pending_fixup(self) -> bool:
        if self._sidecar_has_pending_fixup is None:
            from imagetagger.utils.sidecar import read_sidecar_data
            self._sidecar_has_pending_fixup = read_sidecar_data(self.image_path).has_pending_fixup
        return self._sidecar_has_pending_fixup

    @property
    def sidecar_validated(self) -> str | None:
        if self._sidecar_validated is _UNKNOWN:
            from imagetagger.utils.sidecar import read_sidecar_data
            self._sidecar_validated = read_sidecar_data(self.image_path).validated
        return self._sidecar_validated  # type: ignore[return-value]
