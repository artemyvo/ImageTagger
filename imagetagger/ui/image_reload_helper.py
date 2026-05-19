"""Image file monitoring and reload detection for external editor changes.

Provides centralized file watcher and polling logic for detecting when images
are modified by external editors. Polling fallback keeps reload reliable on
network filesystems (SMB/NFS) where native change notifications can be delayed
or suppressed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable
from PyQt6.QtCore import QFileSystemWatcher, QTimer, QObject


class ImageReloadHelper:
    """Centralizes file watcher and polling logic for image reload detection.
    
    This helper manages:
    - QFileSystemWatcher for native OS file change notifications
    - Debounce timer to coalesce rapid file changes
    - Polling fallback for SMB/NFS network drives
    - mtime tracking to detect modifications
    """

    # Debounce interval (ms): delay before triggering reload after file change
    DEBOUNCE_INTERVAL_MS = 400
    
    # Polling interval (ms): check mtime when watcher events are unreliable
    POLLING_INTERVAL_MS = 1200

    def __init__(
        self,
        parent: QObject,
        on_reload: Callable[[Path], None],
    ):
        """Initialize image reload helper.
        
        Args:
            parent: QObject parent for timer/watcher lifecycle
            on_reload: Callback invoked when image reload should occur
        """
        self._on_reload = on_reload
        self._parent_qobject = parent
        self._watched_image_path: Path | None = None
        self._watched_image_mtime_ns: int | None = None
        self._image_reload_pending = False

        self._image_file_watcher: QFileSystemWatcher | None = QFileSystemWatcher(parent)
        self._image_file_watcher.fileChanged.connect(self._on_watched_image_file_changed)

        self._image_reload_debounce_timer = QTimer(parent)
        self._image_reload_debounce_timer.setSingleShot(True)
        self._image_reload_debounce_timer.setInterval(self.DEBOUNCE_INTERVAL_MS)
        self._image_reload_debounce_timer.timeout.connect(self._apply_pending_image_reload)

        # Polling fallback keeps reload reliable on SMB/NFS volumes where
        # native change notifications can be delayed or missing.
        self._image_reload_poll_timer = QTimer(parent)
        self._image_reload_poll_timer.setInterval(self.POLLING_INTERVAL_MS)
        self._image_reload_poll_timer.timeout.connect(self._poll_watched_image_changes)

    def set_watched_image(self, image_path: Path | None) -> None:
        """Set the image to monitor for external changes.
        
        Args:
            image_path: Path to image to watch, or None to stop watching
        """
        self._image_reload_pending = False
        self._image_reload_debounce_timer.stop()

        if self._image_file_watcher is not None:
            existing_watch_paths = self._image_file_watcher.files()
            if existing_watch_paths:
                self._image_file_watcher.removePaths(existing_watch_paths)

        self._watched_image_path = image_path
        self._watched_image_mtime_ns = None

        if image_path is None:
            self._image_reload_poll_timer.stop()
            # Release the inotify fd immediately rather than waiting for the
            # parent widget to be destroyed.  Qt's DeferredDelete (deleteLater)
            # is not dispatched while a nested modal exec() loop is running at
            # one level deeper than where deleteLater was called, so dialogs
            # shown in the fixup navigation loop accumulate stale
            # QFileSystemWatcher objects — each holding an open inotify fd —
            # until the entire loop exits.  Removing the Qt parent and dropping
            # the Python reference transfers ownership to Python and causes sip
            # to delete the C++ object (and close its fd) synchronously.
            if self._image_file_watcher is not None:
                self._image_file_watcher.setParent(None)
                self._image_file_watcher = None
            return

        # Recreate the watcher if it was previously released.
        if self._image_file_watcher is None:
            self._image_file_watcher = QFileSystemWatcher(self._parent_qobject)
            self._image_file_watcher.fileChanged.connect(self._on_watched_image_file_changed)

        self._watched_image_mtime_ns = self._image_mtime_ns(image_path)
        try:
            if image_path.exists():
                self._image_file_watcher.addPath(str(image_path))
        except OSError:
            pass

        if not self._image_reload_poll_timer.isActive():
            self._image_reload_poll_timer.start()

    def _ensure_watched_image_subscription(self) -> None:
        """Re-attach watcher if subscription was lost.
        
        External editors may do atomic-replace saves; re-subscribe after reload.
        """
        image_path = self._watched_image_path
        if image_path is None or self._image_file_watcher is None:
            return
        current_watches = {
            os.path.normcase(path)
            for path in self._image_file_watcher.files()
        }
        target = os.path.normcase(str(image_path))
        if target in current_watches:
            return
        try:
            if image_path.exists():
                self._image_file_watcher.addPath(str(image_path))
        except OSError:
            pass

    def _schedule_pending_image_reload(self) -> None:
        """Schedule image reload with debounce to handle rapid file changes."""
        self._image_reload_pending = True
        self._image_reload_debounce_timer.start()

    def _on_watched_image_file_changed(self, changed_path: str) -> None:
        """Handle QFileSystemWatcher file change event."""
        image_path = self._watched_image_path
        if image_path is None or self._image_file_watcher is None:
            return

        normalized_changed = os.path.normcase(changed_path)
        normalized_target = os.path.normcase(str(image_path))
        if normalized_changed != normalized_target:
            return

        self._ensure_watched_image_subscription()
        self._schedule_pending_image_reload()

    def _poll_watched_image_changes(self) -> None:
        """Polling fallback: check mtime when watcher events unreliable."""
        image_path = self._watched_image_path
        if image_path is None:
            self._image_reload_poll_timer.stop()
            return

        current_mtime = self._image_mtime_ns(image_path)
        if current_mtime is None:
            return

        if self._watched_image_mtime_ns is None:
            self._watched_image_mtime_ns = current_mtime
            return

        if current_mtime != self._watched_image_mtime_ns:
            self._schedule_pending_image_reload()

    def _apply_pending_image_reload(self) -> None:
        """Apply deferred image reload and invoke callback."""
        if not self._image_reload_pending:
            return

        self._image_reload_pending = False
        image_path = self._watched_image_path
        if image_path is None:
            return

        # External editors may do atomic-replace saves; re-subscribe every reload.
        self._ensure_watched_image_subscription()

        # Update mtime after successful reload
        mtime = self._image_mtime_ns(image_path)
        if mtime is not None:
            self._watched_image_mtime_ns = mtime

        # Invoke caller's reload handler
        self._on_reload(image_path)

    @staticmethod
    def _image_mtime_ns(image_path: Path) -> int | None:
        """Get image file modification time in nanoseconds.
        
        Args:
            image_path: Path to check
            
        Returns:
            Modification time in nanoseconds, or None if unavailable
        """
        try:
            return image_path.stat().st_mtime_ns
        except OSError:
            return None
