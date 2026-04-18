---
description: "Use when editing merge dialog keyboard behavior, shortcut registration, or key event handling."
applyTo: "{imagetagger/merge_dialog.py,imagetagger/shortcuts.py,docs/shortcuts.md}"
---

# Merge Dialog Keystroke and Shortcut Rules

These rules define intended behavior for merge dialog keyboard interaction. Keep behavior stable unless a change is explicitly requested.

## Platform Mapping

- Treat Alt on Win/Linux as the same intent as Option on macOS.
- Keep Cmd-based macOS mappings where already defined for navigation parity.
- Keep shortcut labels and menu/button hints aligned with effective platform mappings.

## Shortcut Intent (Must Preserve)

- Merge and next: Alt+Enter (Alt+Return equivalent); macOS Option+Enter.
- Focus quick-add tag input: Alt+T; macOS Option+T.
- Undo merge/local changes: Ctrl+Z; macOS Command+Z.
- Previous actionable row: Alt+Up; macOS Option+Up.
- Next actionable row: Alt+Down; macOS Option+Down.
- Accept all proposed rows and merge: Alt+A; macOS Option+A.
- Regenerate proposed annotations: Alt+R; macOS Option+R.
- Previous image: Alt+Left; macOS Command+[.
- Next image: Alt+Right; macOS Command+].
- Delete selected existing/current tags: Delete or Backspace, including macOS Fn+Delete behavior where applicable.

## Table Navigation and Editing Behavior (No Dedicated Shortcut Registration)

- Up/Down arrows move the selected comparison row.
- Home selects the first comparison row.
- End selects the last comparison row.
- Left arrow applies the proposed value for the selected row.
- Left arrow row-apply behavior must work even when the comparison table itself does not currently have focus, including after returning to the dialog by clicking non-control regions (for example grey spacer areas, image preview, or title bar).
- Enter/Return triggers the selected row action.
- Delete/Backspace removes selected current rows where deletion is valid.

## Left Arrow Priority and Exceptions

- Left arrow defaults to row-apply intent in the merge dialog (apply proposed value on the selected row).
- Do not steal Left arrow while a dialog text input has focus (for example quick-add QLineEdit or QTextEdit-based editors).
- Do not steal Left arrow while editing a tag/value cell inside the comparison table; in-cell editing keeps normal cursor movement behavior.
- Do not override Left arrow behavior for focused controls that have their own Left-arrow semantics (for example slider/spinbox/other navigable controls).
- Only trigger row-apply when no focused control has a more specific Left-arrow meaning.

## Conflict and Safety Rules

- Do not bind Enter alone to global merge actions; Enter remains row-local unless a specific button/action has focus.
- Merge and next (Alt+Enter/Alt+Return) must be disabled whenever any edit box has focus (for example QLineEdit/QTextEdit) or while a comparison-table cell editor is active, except when focus is in the quick-add tag QLineEdit and the field is empty.
- Do not break text-entry expectations in editable inputs (for example quick-add field navigation and typing).
- Prefer ignoring unknown keys over triggering implicit destructive actions.
- If a shortcut is changed, update all of the following in the same change:
  - Merge dialog action wiring in code.
  - Any button/action labels or tooltips displaying key hints.
  - User documentation in docs/shortcuts.md.

## Accessibility and Consistency

- Keyboard-only flow must remain usable for row review, row acceptance, deletion, undo, and merge navigation.
- Preserve deterministic behavior: one key press should map to one clear action.
- Preserve focus visibility and avoid hidden focus jumps after shortcut execution.

## Validation Checklist

- Verify Win/Linux and macOS mappings are both handled.
- Verify row-level keys (Up/Down/Home/End/Left/Enter/Delete) still behave as intended.
- Verify global dialog shortcuts still invoke the intended actions.
- Verify docs/shortcuts.md matches implementation after any keyboard change.
