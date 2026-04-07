# ImageTagger Keyboard Shortcuts

Navigation: [Project README](../README.md) | [Docs Index](README.md)

This page lists all keyboard shortcuts currently wired in the application code.

## Platform Legend

- Win/Linux: Windows and Linux default mapping.
- macOS: Native macOS mapping.
- `Option` on macOS is the same physical modifier as `Alt`.

## Main Window

| Action | Win/Linux | macOS | Where |
|---|---|---|---|
| Select all visible images | Ctrl+A | Command+A | Image list |
| Jump to first fixup | Alt+F | Option+F | Image list |
| Jump to last fixup | Alt+L | Option+L | Image list |
| Remove selected tag | Delete | Delete (or Fn+Delete on some keyboards) | Tag list |
| Open folder | Ctrl+L | Command+O | File menu |
| Refresh folder | Ctrl+R | Command+R | File menu |
| Exit app | Ctrl+Q | Command+Q | File menu |
| Increase font | Ctrl++ / Ctrl+= | Command++ / Command+= | Edit menu |
| Decrease font | Ctrl+- | Command+- | Edit menu |

## Merge Dialog

| Action | Win/Linux | macOS | Where |
|---|---|---|---|
| Remove selected existing tags | Delete or Backspace | Delete or Backspace (Fn+Delete on some keyboards) | Left/current list |
| Merge and next | Alt+Enter (also Alt+Return) | Option+Enter (also Option+Return) | Dialog action + Merge and Next button |
| Focus quick-add tag input | Alt+T | Option+T | Dialog action |
| Undo merge/local changes | Ctrl+Z | Command+Z | Dialog action + Undo button |
| Previous actionable row | Alt+Up | Option+Up | Dialog action |
| Next actionable row | Alt+Down | Option+Down | Dialog action |
| Regenerate proposed annotations | Alt+R | Option+R | Regenerate button |
| Previous image | Alt+Left | Command+[ | Prev button |
| Next image | Alt+Right | Command+] | Next button |

## Merge Dialog Keyboard Behaviors (No Dedicated Shortcut Registration)

| Behavior | Win/Linux | macOS |
|---|---|---|
| Move row selection | Up/Down arrows | Up/Down arrows |
| Jump to first comparison row | Home | Home, Fn+Left, or Command+Up |
| Jump to last comparison row | End | End, Fn+Right, or Command+Down |
| Apply proposed value for selected row | Left arrow | Left arrow |
| Trigger current row action | Enter/Return | Enter/Return |
| Delete selected current rows | Delete or Backspace | Delete or Backspace (Fn+Delete on some keyboards) |

## Windows vs Linux Differences

No application-defined shortcut differences between Windows and Linux.

Notes:
- Some Linux desktop/window-manager configurations reserve certain `Alt+...` combinations globally.
- If that happens, the shortcut may be intercepted before ImageTagger receives it.
