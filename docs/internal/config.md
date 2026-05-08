# Configuration Reference

Navigation: [Project README](../../README.md) | [Docs Index](../README.md) | [Usage Guide](../usage.md)

All settings are stored in `config.json` in the project root. The file is written automatically by the application; most settings can be changed through the UI. Keys not present in the file fall back to the defaults documented here.

---

## Application State

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `last_open_directory` | string | `""` | Last directory opened by the file browser. Restored on next launch. |
| `last_selected_image` | string | `""` | Path of the last selected image. Restored on next launch. Omitted from the file when empty. |
| `main_window_geometry` | object | `{}` | Saved position and size of the main window (`x`, `y`, `width`, `height`). Empty object means the OS default is used. |
| `merge_dialog_geometry` | object | `{}` | Saved position and size of the merge/fixup dialog. Same structure as `main_window_geometry`. |

---

## UI Appearance

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `font_point_size` | integer | `0` | Application-wide font size in points. `0` means the system default is used. Clamped to 8–40 when applied. |

---

## File Operations

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `confirm_on_delete` | boolean | `true` | When `true`, a confirmation dialog is shown before deleting an image. |
| `directory_loader_max_threads` | integer | `8` | Number of background threads used to load image metadata when opening a directory. Minimum `1`. |

---

## LLM / Provider Settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `llm_endpoint` | string | `""` | Base URL of the LLM server (e.g. `http://localhost:11434` for Ollama). |
| `llm_model` | string | `""` | Model name to request from the server. |
| `llm_max_resolution_mpx` | number | `5.0` | Maximum resolution (in megapixels) to which images are downscaled before being sent to the model. Minimum `0.01`. |
| `llm_threads` | integer | `1` | Number of parallel inference requests. Set to `0` to enable auto mode (see [Ollama Settings](ollama_settings.md)). Minimum `0`. |

### Auto Thread Mode

These keys are used only when `llm_threads` is `0`. See [Ollama Settings](ollama_settings.md) for full details and presets.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `llm_auto_max_threads` | integer | `48` | Hard upper limit on the number of threads auto mode may use. Minimum `1`. |
| `llm_auto_warmup_items` | integer | `4` | Minimum number of completions that must finish before auto mode considers scaling up. Minimum `0`. |
| `llm_auto_scale_up_every` | integer | `3` | How many healthy measurement windows must pass between consecutive scale-up steps. Minimum `1`. |

---

## Merge Dialog Mouse Actions

Stored as a nested object under `merge_table_mouse_actions`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `double_click_action_enabled` | boolean | `true` | When `true`, double-clicking a row in the merge table performs the row action. |
| `swipe_actions_enabled` | boolean | `false` | When `true`, touch/trackpad swipe gestures trigger row actions in the merge table. |
| `horizontal_scroll_actions_enabled` | boolean | `false` | When `true`, horizontal scroll events on the merge table trigger row actions. |
| `horizontal_scroll_reverse_enabled` | boolean | `false` | Reverses the direction mapping for horizontal scroll actions. |
| `horizontal_scroll_stop_idle_seconds` | number | `0.45` | Seconds of scroll inactivity after which the horizontal scroll action is considered complete. Minimum `0.0`. |
| `horizontal_scroll_row_target_mode` | integer | `3` | Controls which row a horizontal scroll event targets. `1` = row under pointer, `2` = selected row, `3` = row under pointer when it matches the selected row. |

---

## Agent Roles

Stored as a nested object under `agent_roles`. Each key is an arbitrary role name and the value is a system-prompt string prepended to the matching LLM request.

| Key | Type | Description |
|-----|------|-------------|
| `description` | string | System prompt used when generating image descriptions. |
| `tagging` | string | System prompt used when generating image tags. |

Additional keys are accepted and ignored unless a matching prompt template references them.

**Example:**

```json
"agent_roles": {
  "description": "You are an expert in nature photography.",
  "tagging": "You are an expert in nature photography."
}
```

---

## Debugging

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `debug_regenerate_prompt_console` | boolean | `false` | When `true`, prints the full prompt sent to the LLM during regeneration to the console. Useful for prompt debugging. |
