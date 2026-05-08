# Sidecar File Reference

Navigation: [Project README](../../README.md) | [Docs Index](../README.md) | [Usage Guide](../usage.md)

Each image managed by ImageTagger has an optional companion JSON file with the same base name and a `.json` extension (e.g. `photo.jpg` → `photo.json`). The file is created automatically when tags, descriptions, or fixup data are first saved. All keys are optional unless noted; absent keys are treated as empty/`null`.

---

## Committed Fields

These fields hold the accepted, finalized output that appears in the main window.

| Key | Type | Description |
|-----|------|-------------|
| `description` | string | Accepted image description. Written when the user saves from the main window or accepts a fixup. Always present in the file (may be an empty string). |
| `reasoning` | string | Chain-of-thought text produced by the LLM alongside the description (from the `THOUGHT:` section of the Vision prompt response). Always present in the file (may be an empty string). |

### VLM Training Dataset Extraction

`description` and `reasoning` are the two fields intended for building a VLM fine-tuning dataset. Both fields are **always present** in the file (they may be empty strings, but the keys are never omitted).

- `description` is the high-density visual caption produced by the **Vision prompt** — it is *not* the short diffusion-model caption stored in the `.txt` sidecar. The two serve different purposes and are kept entirely separate.
- `reasoning` is the chain-of-thought trace that preceded it. It is stored specifically to support **thinking-enabled VLMs** (e.g. models trained with reasoning traces such as Qwen3-VL). Including the CoT in training teaches the model to reason before answering, not just to reproduce captions.

Because VLM-targeted descriptions tend to be denser and more accurate than a direct single-pass diffusion caption, ImageTagger introduces a **Refine** step: the Vision output (`description` + `reasoning`) is fed back into a second prompt that distils it into structured diffusion tags and a short caption. This positive-feedback loop is why Refine-generated diffusion annotations are typically of higher quality than generating tags and description in isolation.

Extraction is straightforward — every `.json` sidecar in the folder has the same flat structure. A minimal conversion script only needs to iterate the directory, read each file, and map `description` + `reasoning` to whatever format your training framework expects (e.g. Unsloth's ShareGPT or Alpaca JSON schemas).

---

## Pending Fixup Fields

These fields are written by the Validate / Generate workflow and consumed by the merge dialog. They are cleared after a fixup is accepted or rejected. All are omitted from the file when absent (`null`).

| Key | Type | Description |
|-----|------|-------------|
| `fixup_issues` | string | Free-text summary of issues found during validation (the `ISSUES:` section of the LLM response). Drives the left-hand panel of the merge dialog. |
| `fixup_tags` | array of strings | Corrected tag list suggested by the LLM (the `TAGS:` section). Shown in the merge table for diff review. |
| `fixup_description` | string | Corrected description suggested by the LLM (the `DESCRIPTION:` section). Shown alongside the existing description for comparison. |
| `fixup_model` | string | Name of the model that produced the pending fixup. Recorded for informational purposes; cleared or updated when a new fixup is written. |
| `fixup_date` | string | ISO-8601 timestamp of when the pending fixup was generated. Cleared or updated together with `fixup_model`. |

---

## AI Find Fields

Written by the AI Find feature. Cleared when a fixup is fully accepted.

| Key | Type | Description |
|-----|------|-------------|
| `ai_find_matches` | array of strings | Normalized search queries for which the AI determined this image is a match. Each entry is a lowercase, normalized version of the original query string. Preserved across validation fixup cycles; only cleared by a full accept/reject of the fixup dialog. |

---

## Vision / Refine Fields

Written by the Refine (vision comparison) workflow. Cleared when a fixup is accepted.

| Key | Type | Description |
|-----|------|-------------|
| `vision_tags` | array of strings | Tag list produced by the refine pass (the `VISIONTAGS:` section). Shown in the comparison panel for side-by-side review against the committed tags. |
| `vision_caption` | string | Caption produced by the refine pass (the `VISIONDESC:` section). Shown alongside the committed description in the comparison panel. |

---

## Validation Stamp Fields

Set when an image passes validation. Both fields are written together and are omitted when absent.

| Key | Type | Description |
|-----|------|-------------|
| `validated` | string | ISO-8601 UTC timestamp of when the image was last validated successfully (e.g. `2026-05-08T14:32:00Z`). |
| `validated_by` | string | Who performed the validation. `"user"` when the user resolved issues manually via the merge dialog; the model name (e.g. `"gemma-4:12b"`) when the model returned "ok" with no issues. |

The `validated` field is set in two situations:

1. **Model returned OK** — the Validate workflow receives an "ok" response; `clear_validation_fields_sidecar` is called, which sets `validated` and `validated_by = <model name>`.
2. **User resolved via merge dialog** — the user completes a merge; `clear_fixup_sidecar` is called, which sets `validated` and `validated_by = "user"`.

The image preview shows a tooltip **"Validated by `<user|model_name>` on `<date>`"** when `validated` is present.

The `✅` badge in the file list is shown when `validated` is present, regardless of how it was set.

---

## Pending Fixup Detection

An image is considered to have a **pending fixup** (shown with a `⚖️` badge in the file list) when any of the following keys is present and non-empty:

- `fixup_issues`
- `fixup_tags`
- `fixup_description`
- `ai_find_matches`
- `vision_tags`
- `vision_caption`

`fixup_model` and `fixup_date` alone do **not** trigger the pending state.

The `✅` badge is shown when `validated` is present and non-null.

---

## Example

```json
{
  "description": "A woman in a red dress stands in a sunlit doorway.",
  "reasoning": "The scene is a backlit interior threshold shot...",
  "fixup_issues": "Tags missing fabric detail; description too brief.",
  "fixup_tags": ["woman", "red dress", "doorway", "backlit", "interior"],
  "fixup_description": "A woman in a flowing red dress stands framed by a sunlit doorway, rim-lit by diffuse afternoon light.",
  "fixup_model": "gemma-4:12b",
  "fixup_date": "2026-05-08T14:32:00Z",
  "ai_find_matches": ["red dress doorway"],
  "vision_tags": ["woman", "red dress", "doorway", "backlit"],
  "vision_caption": "A female subject in crimson fabric occupies a threshold space.",
  "validated": "2026-05-08T15:10:00Z",
  "validated_by": "user"
}
```
