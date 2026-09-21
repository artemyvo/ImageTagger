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
    # Stored pixel size (width, height) as read by the folder loader; None when
    # the file could not be opened.  Kept in the file's own pixel grid (no EXIF
    # orientation applied), the grid the ratio crop works in.
    _image_size: tuple[int, int] | None = field(default=None, repr=False)
    # Cached "size fits none of the allowed ratios" verdict for _image_size;
    # None until MainWindow._ratio_fix_needed has computed it.
    _ratio_fix_needed: bool | None = field(default=None, repr=False)
    # Cached Autofix crop for _image_size (a CropCandidate or None);
    # _UNKNOWN until MainWindow._ratio_autofix_candidate has computed it.
    _ratio_autofix: object = field(default_factory=lambda: _UNKNOWN, repr=False)
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
