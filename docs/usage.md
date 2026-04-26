# ImageTagger Usage Guide

Navigation: [Project README](../README.md) | [Docs Index](README.md)

This guide covers day-to-day usage, workflow details, and project behavior.

## Who This Is For

- ML engineers maintaining image-caption or image-tag datasets.
- Researchers running iterative dataset cleanup before training or fine-tuning.
- Synthetic data and LoRA workflow builders who need fast human-in-the-loop correction.
- Small teams that prefer local desktop workflows over heavier annotation platforms.

## Core Concepts

ImageTagger works with image and sidecar text pairs.

- Each image is associated with a text file that has the same base name.
- Example pair:
  - photo01.jpg
  - photo01.txt
- If the text file does not exist, it is created on save.

## Main Workflow

1. Open a folder with images.
2. If needed, install Ollama from https://ollama.com.
3. Connect to an Ollama server and select a model.
4. Select one or more images in the left panel.
5. Run Generate, Validate, AI Find, and Fixup as needed.

## Image Context Menu

Right-click the image preview in both the main window and merge dialog to open image actions.

- Open in Default App
- Open With (detected editors or custom executable)
- Delete file

Delete file removes:

- image file
- matching .txt sidecar
- matching .fixup files (both preferred and legacy naming)

In the main window, after delete, selection moves to the next image; if deleted item was last, selection moves to the new last image.

In the merge dialog, deleting the current file proceeds to the next fixup file when available. If no fixup files remain, the merge table is cleared, image preview is disabled, action buttons are disabled, and you can close the dialog with Esc.

## Ollama and Model Recommendations

- Ollama is the local model runtime used by ImageTagger. If you do not already have it, install it from https://ollama.com.
- Recommended model: Qwen3-VL-8B.
- Current practical recommendation: start with Qwen3-VL-8B for the best observed performance/quality balance.
- Thread count defaults to 1 for safety and to avoid overcommitting slower GPUs.
- Setting thread count to 0 enables auto mode, which rebalances thread count dynamically and usually produces good results once it settles.
- On RTX 3090-class and newer top-end GPUs, Qwen3-VL-8B can often scale up to 16 threads with a significant speedup compared to a single thread.
- The best thread count depends on your GPU, VRAM pressure, and what else is running on the machine, so it is worth experimenting.
- During Generate, Validate, and AI Find tasks, the current thread count is shown in the status line.

## AutoTag Operations

### Generate

Generate adds tags and/or description to selected images.

- Use checkboxes to choose Tags and Description.
- Timeout is a per-image budget.
- Retries can be configured.
- Downscale controls image query resolution before sending to Ollama.
- Threads can be fixed or set to 0 for auto behavior.
- Default is 1 thread for safety; on strong GPUs, especially RTX 3090-class and newer hardware running Qwen3-VL-8B, testing higher values up to 16 can produce substantial speedups.
- Setting threads to 0 enables automatic balancing; after it settles, it usually gives good results and the live thread count is visible in the status line while a task is running.

### Validate

Validate checks existing annotations and writes fixup files when needed.

- Runs on selected images with existing annotations.
- Creates one .fixup file per image with detected issues.
- Removes stale .fixup files when validation result is OK.

### AI Find

AI Find checks whether selected images contain a target concept.

- Enter a concept in the AI Find field.
- Run on selected images.
- Matching images are tracked and recorded in fixup/search data.

### Fixup

Fixup opens the merge dialog for the current image when a fixup exists.

- Review proposed description and tag changes.
- Apply accept, reject, merge, and next/previous navigation actions.
- Regenerate can be run from inside the fixup dialog with its own controls.
  - Useful when existing tags and description are completely messed up; regenerate fresh candidates and compare side-by-side before merging.

### Merge Dialog Regenerate Overrides

Inside the merge dialog, regeneration has local controls that can override your main-window defaults for the current fixup pass:

- Server URL input, Fetch models, model dropdown, and Use button let you switch regenerate calls to a different Ollama/OpenAI-compatible endpoint and model.
- Description prompt and Tags prompt tabs let you locally edit prompt text used by regenerate.
- These overrides are scoped to merge-dialog regenerate behavior and do not replace your main-window model selection.

This is especially useful when a model struggles to regenerate good tags or description with its default settings. You can test a stronger model, a different endpoint, or stricter local prompt wording immediately, then compare results side-by-side before merging.

### Merge Dialog Interface

The merge dialog presents a meld-like 2-way comparison:

- Left pane: current (existing) annotation.
- Right pane: proposed (fixup) annotation.

Navigation:

- Arrow keys: Move through comparison rows inside the table.
- Alt+Up / Alt+Down: Jump to previous/next difference row.

Quick Actions:

- Alt+T: Quick add a new tag not present in existing rows.
- Alt+A: Accept all proposed rows and merge.
- Alt+R: Start regeneration to create fresh candidates.
- Alt+Enter: Merge current change (save left/current pane to image and proceed to next image).
- Left arrow key: Accept proposed change from right into result.
- Del key: Delete selected row.

Use Merge/Reject buttons to apply your final decision and navigate to the next fixup image.

### Merge Dialog Mouse Actions

The merge comparison table also supports mouse and trackpad actions:

- Double-click (left button): triggers the current row action (apply proposed value, delete current value, or add suggested tag depending on row/action state).
- Right-click opens row context menu.
- On macOS trackpads, a two-finger tap maps to right-click and opens the same row context menu.
- Horizontal swipe/drag can trigger row actions when enabled.
- Horizontal scroll (mouse horizontal wheel or trackpad horizontal two-finger scroll) can trigger row actions when enabled.

Defaults:

- Double-click action: enabled.
- Swipe actions: disabled.
- Horizontal scroll actions: disabled.
- Horizontal scroll reverse: disabled.
- Horizontal scroll stop-idle seconds: 0.45.
- Horizontal scroll row-target mode: 3 (safest mode).

Configuration is stored in config.json under `merge_table_mouse_actions`:

```json
"merge_table_mouse_actions": {
  "double_click_action_enabled": true,
  "swipe_actions_enabled": false,
  "horizontal_scroll_actions_enabled": false,
  "horizontal_scroll_reverse_enabled": false,
  "horizontal_scroll_stop_idle_seconds": 0.45,
  "horizontal_scroll_row_target_mode": 3
}
```

Possible values:

- `double_click_action_enabled`: `true` or `false`.
- `swipe_actions_enabled`: `true` or `false`.
- `horizontal_scroll_actions_enabled`: `true` or `false`.
- `horizontal_scroll_reverse_enabled`: `true` or `false`.
- `horizontal_scroll_stop_idle_seconds`: number `>= 0`.
  - `0` disables stop-before-rearm gating, so long continuous scrolling can trigger repeated actions.
  - `> 0` requires scrolling to stop/idle for that many seconds before another scroll action can trigger.
- `horizontal_scroll_row_target_mode`: `1`, `2`, or `3`.
  - `1`: action applies to row under mouse pointer.
  - `2`: action applies to selected row regardless of pointer position.
  - `3`: action applies only when pointer is over selected row (default).

## Filter Syntax

Use the image list filter to narrow large datasets quickly.

Supported terms:

- `fixup`: images with a fixup file.
- `untagged`: images that have no annotation (.txt) file at all.
- `resolution <, >, <=, >=`: images matching a resolution threshold in megapixels.
- `"tag"`: exact tag match.
- `'text'`: case-insensitive text match against annotation content.

Operators:

- `!` or `~`: NOT (negates the following term or group)
- `&`: AND
- `|`: OR
- `( ... )`: grouping

Precedence (highest to lowest): NOT, AND, OR — same as C. So `a | b & c` is `a | (b & c)` and `!a & b` is `(!a) & b`.

Examples:

- `!fixup` — images without fixup files
- `resolution < 1.0` — images with resolution lower than 1 MPx
- `resolution >= 5` — images with resolution 5 MPx or higher
- `(resolution > 5) & 'landscape'` — high-res landscape images
- `fixup & "portrait"`
- `!"landscape" | 'sunset'`
- `~fixup & ("animal" | 'night')`

## Keyboard Shortcuts

See the full cross-platform shortcut reference:

- [shortcuts.md](shortcuts.md)

## Prompt Files

Prompt files are optional and loaded from the prompts directory in the project root. If missing, built-in defaults are used.

- prompts/description_prompt.txt
- prompts/tags_prompt.txt
- prompts/validation_prompt.txt
- prompts/search_prompt.txt

In the app, prompt tabs allow in-memory apply, save to file, and reset to default behavior.

## Screenshots

Main window:

![ImageTagger Main Window](screenshots/main_window.png)

Merge dialog:

![ImageTagger Merge Dialog](screenshots/merge_dialog.png)

## Platform Notes

ImageTagger is built with Python and PyQt6.

Python 3.9 or newer is required.

Windows, Linux, and macOS are tested.

Install and run scripts are provided for all three platforms:

- Windows: `install.bat`, `run.bat`, `update.bat`
- Linux / macOS: `install.sh`, `run.sh`, `update.sh`

## Configuration and Persistence

config.json stores session and UI state, including:

- last opened directory
- selected Ollama server and model
- query downscale value (llm_max_resolution_mpx)
- thread setting
- window geometry state
- merge-dialog mouse action settings in `merge_table_mouse_actions`
- `confirm_on_delete` (default `true`): show confirmation dialog before deleting from image context menu

For a full list of Ollama and auto-mode keys, see [ollama_settings.md](ollama_settings.md).

## Acknowledgement and Inspiration

This project is heavily inspired by TagGUI.

TagGUI deserves full credit for the core layout direction and practical workflow ideas that informed this project.
