# Pixel Purge

Fully local-first Google Photos cleanup, deduplication, and classification engine.
No cloud hosting — everything runs on your machine. See [`docs/pixel-purge-prd.md`](docs/pixel-purge-prd.md).

## Status

**Phase 1** (in progress): local Takeout ingestion + hierarchical deduplication CLI.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # core
pip install -e '.[heic]'    # + HEIC (iPhone) support
pip install -e '.[raw]'     # + camera RAW support
pip install -e '.[dev]'     # + test tooling
```

## Usage (Phase 1)

```bash
pixel-purge init                          # create ~/.pixel-purge/manifest.db
pixel-purge ingest ~/Downloads/Takeout/   # discover media + merge metadata
pixel-purge ingest photos.tgz             # or from a .tgz archive
pixel-purge dedup                         # run all 3 dedup tiers
pixel-purge dedup --tier 1                # exact-hash only
pixel-purge dedup --stats                 # show statistics
```

Duplicates are **flagged for review, never auto-deleted** — deletion is a
separate, human-gated step (Module D, later phase).

## Development

```bash
pip install -e '.[dev]'
pytest
```
