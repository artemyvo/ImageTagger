from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class ImageRecord:
    image_path: Path
    text_path: Path
    text: str
    _sidecar_has_pending_fixup: bool | None = field(default=None, repr=False)
    _resolution_mpx: float | None = field(default=None, repr=False)

    @property
    def has_pending_fixup(self) -> bool:
        if self._sidecar_has_pending_fixup is None:
            from imagetagger.utils.sidecar import read_sidecar_data
            self._sidecar_has_pending_fixup = read_sidecar_data(self.image_path).has_pending_fixup
        return self._sidecar_has_pending_fixup
