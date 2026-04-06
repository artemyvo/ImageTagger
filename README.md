# ImageTagger

ImageTagger is a desktop annotation tool for image/text pairs, built with PyQt6 for ML dataset curation workflows.

It is designed for teams and solo practitioners who need to keep large caption/tag corpora clean, consistent, and model-ready.

## Quick Start

Windows PowerShell:

1. python -m venv .venv
2. .\.venv\Scripts\Activate.ps1
3. pip install -r requirements.txt
4. python run.py

Linux (bash):

1. python3 -m venv .venv
2. source .venv/bin/activate
3. pip install -r requirements.txt
4. python run.py

Then:

1. Open a folder containing images.
2. If you do not already have Ollama, install it from https://ollama.com.
3. Connect to Ollama and choose a model.
4. Qwen3-VL-8B is the current recommended model for the best practical performance/quality balance.
5. Select one or more images.
6. Generate, validate, and resolve fixups.

Supported image formats: jpg, jpeg, png, bmp, gif, webp.

## Documentation

- Docs index: [docs/README.md](docs/README.md)
- Full usage guide: [docs/usage.md](docs/usage.md)
- Ollama settings: [docs/ollama_settings.md](docs/ollama_settings.md)
- Screenshots: [docs/screenshots/main_window.png](docs/screenshots/main_window.png), [docs/screenshots/merge_dialog.png](docs/screenshots/merge_dialog.png)

## Highlights

- Batch operations on selected images for Generate, Validate, and AI Find.
- Fixup workflow backed by per-image .fixup files.
- Expression-based filtering with fixup, quoted tags, freetext, AND/OR, and parentheses.
- Editable prompt tabs for Description, Tagging, Validation, and AI Search.
- Preview context menu with Open in Default App and Open With support.

## Screenshots

Main window:

![ImageTagger Main Window](docs/screenshots/main_window.png)

Merge dialog:

![ImageTagger Merge Dialog](docs/screenshots/merge_dialog.png)

## Acknowledgement and Inspiration

This project is heavily inspired by TagGUI.

TagGUI deserves full credit for the core UI layout direction and practical workflow ideas.

## AI Generation Disclosure

For transparency and legal disclosure, this codebase was 100% AI-generated with GitHub Copilot.
