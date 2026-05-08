# ImageTagger Usage Guide

Navigation: [Project README](../README.md) | [Docs Index](README.md)

This guide covers day-to-day usage, workflow details, and project behavior.

## Who This Is For

- ML engineers maintaining image-caption or image-tag datasets.
- Researchers running iterative dataset cleanup before training or fine-tuning.
- Synthetic data and LoRA workflow builders who need fast human-in-the-loop correction.
- Vision language model (VLM) trainers who need dense, reasoning-grounded captions alongside diffusion-style tags.
- Small teams that prefer local desktop workflows over heavier annotation platforms.

## Core Concepts

ImageTagger works with image and sidecar text pairs.

- Each image is associated with a text file that has the same base name.
- Example pair:
  - photo01.jpg
  - photo01.txt
- If the text file does not exist, it is created on save.

ImageTagger distinguishes two annotation purposes:

- **Tags + Description** (Tags tab) — comma-separated tags and a short caption for **diffusion model training** (e.g., LoRA/SDXL/Flux datasets). This is the primary workflow.
- **Vision description** (Vision tab) — a high-density, reasoning-grounded caption designed for **vision language model (VLM) training**. Stored alongside a chain-of-thought (CoT) field. Intended as a ready-to-export source for tools like [Unsloth](https://github.com/unslothai/unsloth) VLM fine-tuning datasets.

## Image List Badges

Each image in the file list can show one or more status badges. Badges indicate which data pipeline sources have contributed pending data to that image:

| Badge | Meaning | Source |
|---|---|---|
| ⚖️ | Fixup pending — the image has AI-proposed tag or description corrections waiting for review | Written by **Validate** |
| ✨ | Vision/Refine data available — the image has a vision caption or refine tags ready for comparison in the merge dialog | Written by **Generate** (with Refine enabled) |
| 🔍 | AI Find match — at least one AI Find query matched this image | Written by **AI Find** |
| ✅ | Validated — the image passed its last validation pass with no outstanding issues | Set by **Validate** (model returned OK) or by the user completing a merge in the **Fixup** dialog |

Badges reflect the current state of the image's `.json` sidecar. They update automatically after each operation.

**Validated tooltip.** Hovering over a ✅-badged image in the list shows a tooltip with the validation date and source, for example: *"Validated by Qwen3-VL-8B on 2026-05-08"* or *"Validated by user on 2026-05-08"*. This lets you see at a glance whether a human or a model signed off on the annotation.

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
- matching .json sidecar

In the main window, after delete, selection moves to the next image; if deleted item was last, selection moves to the new last image.

In the merge dialog, deleting the current file proceeds to the next image with pending fixup data when available. If no pending fixups remain, the merge table is cleared, image preview is disabled, action buttons are disabled, and you can close the dialog with Esc.

## Ollama and Model Recommendations

- Ollama is the local model runtime used by ImageTagger. If you do not already have it, install it from https://ollama.com.
- Recommended model: Qwen3-VL-8B.
- Current practical recommendation: start with Qwen3-VL-8B for the best observed performance/quality balance.
- Thread count defaults to 1 for safety and to avoid overcommitting slower GPUs.
- Setting thread count to 0 enables auto mode, which rebalances thread count dynamically and usually produces good results once it settles.
- On RTX 3090-class and newer top-end GPUs, Qwen3-VL-8B can often scale up to 16 threads with a significant speedup compared to a single thread.
- The best thread count depends on your GPU, VRAM pressure, and what else is running on the machine, so it is worth experimenting.
- During Generate, Validate, and AI Find tasks, the current thread count is shown in the status line.

## Global Tags Panel

The **Tags** tab in the bottom-right controls panel gives a dataset-wide view of every tag used across all images.

![Main window — Global Tags panel](screenshots/mainwindow-tags.png)

- Each entry shows the tag name and the number of images it appears in, for example `natural light (26)`.
- A **Filter tags…** input at the top lets you search the list by substring — useful in large datasets.
- **Multi-selection** is supported via Shift-click (range) and Ctrl-click (individual). Select any number of tags, then press Delete or Backspace to **purge** them from the entire dataset in one operation. A confirmation dialog is shown before any files are written.

Purging is a destructive, dataset-wide write — it removes the selected tags from every `.txt` sidecar that contains them. Use the filter to verify a tag is genuinely unwanted before purging.

## AutoTag Operations

### Generate

Generate adds tags, description, and/or vision annotations to selected images.

- Use checkboxes to choose Tags, Description, and/or Refine (vision).
- Timeout is a per-image budget.
- Retries can be configured.
- Downscale controls image query resolution before sending to the model.
- Threads can be fixed or set to 0 for auto behavior.
- Default is 1 thread for safety; on strong GPUs, especially RTX 3090-class and newer hardware running Qwen3-VL-8B, testing higher values up to 16 can produce substantial speedups.
- Setting threads to 0 enables automatic balancing; after it settles, it usually gives good results and the live thread count is visible in the status line while a task is running.

### Vision Tab

The **Vision tab** (top-right panel, next to Tags) shows the committed vision description and chain-of-thought (CoT) reasoning for the selected image.

This annotation is **separate from the diffusion-model description** stored in the `.txt` sidecar. The two serve entirely different purposes and are never mixed:

- The `.txt` sidecar holds short comma-separated tags and a concise caption for diffusion training.
- The `.json` sidecar `description` + `reasoning` fields hold the high-density VLM caption and its chain-of-thought trace, intended for VLM fine-tuning.

Because VLM-targeted descriptions are usually of higher quality and richer in detail than a direct diffusion caption, this is precisely why the **Refine** step exists: it feeds the Vision output back into a second prompt that distils it into diffusion-friendly tags and a short caption, producing better results than generating diffusion annotations in isolation.

Both `description` and `reasoning` are always present in the `.json` file (may be empty strings). The CoT (`reasoning`) is stored specifically to support **thinking-enabled VLMs** such as Qwen3-VL, where including the reasoning trace in training teaches the model to reason before answering. The intended export path is directly into a [Unsloth](https://github.com/unslothai/unsloth) fine-tuning dataset or any other VLM training pipeline that expects a caption + CoT pair.

- The Vision description is generated by the **Vision prompt** (customisable in the Prompts panel, Vision tab).
- The **Refine** checkbox on the Generate panel triggers vision generation alongside tags/description in the same batch pass.
- Vision description and CoT are editable directly in the tab and saved with the **Save** button.

### Validate

Validate checks existing annotations and writes fixup data into the image's `.json` sidecar when needed.

- Runs on selected images with existing annotations.
- Writes fixup fields (`fixup_issues`, `fixup_tags`, `fixup_description`) into the sidecar for each image with detected issues.
- Clears fixup fields from the sidecar when validation result is OK.

**Example — validation correcting a wrong tag:**

![Merge dialog — Monk Vulture validation example](screenshots/mergedialog-merge.png)

`Aegypius_monachus_-_1.jpg` (a Monk Vulture) was tagged as `eagle`. Validation caught this: the issues banner reads *"The tag 'eagle' is slightly inaccurate as the bird is clearly a vulture (likely a Black Vulture given the dark head and white wingtips)."* In the comparison table, `eagle` appears in the Current column marked for deletion, and `vulture` appears in the Proposed column as the replacement. Accepting the row removes `eagle` and inserts `vulture` in one action.

### AI Find

AI Find checks whether selected images contain a target concept.

- Enter a concept in the AI Find field.
- Run on selected images.
- Matching images are tracked and recorded in the sidecar; matched entries appear with a 🔍 badge in the image list and as rows in the merge dialog.

**AI Find results require manual review.** The model can produce false positives, especially for visually similar species, breeds, or object classes. In the same screenshot above, a search for `milvus migrans` (Black Kite) returned a false positive — the AI incorrectly associated the large brown raptor with a Black Kite. The `milvus migrans` row appears at the bottom of the merge dialog table with a 🔍 icon. Reject it to remove the erroneous match from the sidecar. Never accept AI Find matches wholesale without checking the image.

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
- Del key: Delete selected current row, including description rows when present.

Use Merge/Reject buttons to apply your final decision and navigate to the next fixup image.

### Merge Dialog Mouse Actions

The merge comparison table also supports mouse and trackpad actions:

- Double-click (left button) on an editable Current cell opens in-cell editing.
- Double-click (left button) elsewhere in the row triggers the current row action (apply proposed value, delete current value, or add suggested tag depending on row/action state).
- Right-click opens row context menu.
- On macOS trackpads, a two-finger tap maps to right-click and opens the same row context menu.
- Horizontal swipe/drag can trigger row actions when enabled.
- Horizontal scroll (mouse horizontal wheel or trackpad horizontal two-finger scroll) can trigger row actions when enabled. This is the recommended way to merge or delete rows with a trackpad or a mouse with a horizontal scroll wheel, such as the Logitech MX Master 3. Enable `horizontal_scroll_actions_enabled` to use it.

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

- `fixup`: images with pending fixup data in their sidecar.
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

- `!fixup` — images without pending fixup data
- `resolution < 1.0` — images with resolution lower than 1 MPx
- `resolution >= 5` — images with resolution 5 MPx or higher
- `(resolution > 5) & 'landscape'` — high-res landscape images
- `fixup & "portrait"`
- `!"landscape" | 'sunset'`
- `~fixup & ("animal" | 'night')`

## Keyboard Shortcuts

See the full cross-platform shortcut reference:

- [shortcuts.md](shortcuts.md)

## Prompt Customisation

Every workflow in ImageTagger is driven by an editable prompt. You can customise any of them directly in the app without restarting.

### Prompt Tabs

The **Prompts** panel (bottom-right) has a tab for each workflow:

| Tab | Prompt file | Purpose |
|---|---|---|
| Tags | `prompts/tags_prompt.txt` | Comma-separated tags for diffusion training |
| Description | `prompts/description_prompt.txt` | Short caption for diffusion training |
| Validation | `prompts/validation_prompt.txt` | Annotation quality check |
| Search | `prompts/search_prompt.txt` | AI Find concept search |
| Vision | `prompts/vision_prompt.txt` | High-density VLM training caption with CoT |
| Refine | `prompts/refine_prompt.txt` | Structured tag/caption distillation from Vision output |

Prompt files are loaded from the `prompts/` directory in the project root. If a file is missing, the built-in default is used automatically.

### Editing Prompts

Each prompt tab has four buttons:

- **Apply** — activates the edited text in-memory for the current session without saving to disk.
- **Save** — writes the text to the corresponding file in `prompts/`.
- **Reset** — restores the built-in default (in memory; save to persist).
- **Test** — runs the current prompt against the selected image using the active model and shows a dialog with the full rendered prompt and the raw model response. Use this to debug prompt wording before committing it to a full batch run.

### Agent Role ("You are ...")

Each prompt tab has an **Agent role** input field. Whatever you type there is injected at the top of the prompt as the first line (e.g., `You are a professional image captioner specialising in wildlife photography.`). This sets the model's persona for that specific workflow.

The `{agent_role}` placeholder in the prompt text marks where the role line is inserted. If the field is left blank, the placeholder line is silently removed from the final prompt.

Agent roles are saved per-workflow in `config.json` under the `agent_roles` key.

### Positive Feedback: `{existing_tags}` and `{tags}`

The default Tags and Description prompts include an `{existing_tags}` clause. When the selected image already has committed tags, those tags are injected into the prompt as confirmed seed facts before the model runs. This positive feedback loop allows the model to:

- generate tags that are complementary to what is already there rather than duplicating them (Tags workflow), and
- treat confirmed tags as ground truth when composing the caption (Description workflow), producing more accurate and grounded output.

The Vision prompt uses the same idea via a `{tags}` clause: confirmed tags are passed as authoritative identity facts, which anchors the model's dense caption to the known subject.

### Refine: Vision → Tags/Description Positive Feedback

The **Refine** workflow is a second-pass distillation step that closes the loop between the VLM-targeted Vision output and diffusion annotation:

1. The **Vision** prompt generates a high-density description and chain-of-thought reasoning grounded in the image. This output is *not* a diffusion caption — it is a richer, more analytical description intended for VLM training.
2. The **Refine** prompt receives that vision output as `{vision_data}` and distils it into a structured tag list and a concise diffusion caption.

Because VLM-targeted descriptions are typically denser and more accurate than what a single-pass diffusion prompt produces, feeding them into Refine yields diffusion annotations of measurably higher quality. This positive-feedback loop — Vision reasons deeply about the image, Refine extracts the facts — is the primary motivation for the two-step workflow. Enable the **Refine** checkbox in the Generate panel to run both passes in one batch operation.

## Screenshots

Main window:

![ImageTagger Main Window](screenshots/mainwindow-tags.png)

Merge dialog:

![ImageTagger Merge Dialog](screenshots/mergedialog-merge.png)

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

For a full list of Ollama and auto-mode keys, see [ollama_settings.md](internal/ollama_settings.md).

## Acknowledgement and Inspiration

This project is heavily inspired by TagGUI.

TagGUI deserves full credit for the core layout direction and practical workflow ideas that informed this project.
