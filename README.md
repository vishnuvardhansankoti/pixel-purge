# Pixel Purge

Fully local-first Google Photos cleanup, deduplication, and classification engine.
No cloud hosting — everything runs on your machine. See [`docs/pixel-purge-prd.md`](docs/pixel-purge-prd.md).

## Status

- **Phase 1** (done): local Takeout ingestion + hierarchical deduplication CLI.
- **Phase 2** (done): local AI vision (CLIP zero-shot) + face clustering (InsightFace/DBSCAN) + TUI review.
- **Phase 3** (done): cleanup execution — deletion-manifest export (primary), curation + staging with format-aware metadata restore, gated clean-slate re-upload, experimental browser deletion.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # core (ingest + dedup)
pip install -e '.[heic]'    # + HEIC (iPhone) support
pip install -e '.[raw]'     # + camera RAW support
pip install -e '.[vision]'  # + CLIP zero-shot classification (torch, open_clip)
pip install -e '.[faces]'   # + InsightFace + DBSCAN (onnxruntime, scikit-learn)
pip install -e '.[dev]'     # + test tooling
```

The `vision`/`faces` extras are only needed to actually run `classify`; everything
else (and the whole test suite) works without them.

## Usage

```bash
pixel-purge init                          # create ~/.pixel-purge/manifest.db
pixel-purge ingest ~/Downloads/Takeout/   # discover media + merge metadata
pixel-purge ingest photos.tgz             # or from a .tgz archive
pixel-purge dedup                         # run all 3 dedup tiers
pixel-purge dedup --tier 1                # exact-hash only
pixel-purge dedup --stats                 # dedup statistics

pixel-purge classify                      # CLIP vision + face clustering
pixel-purge classify --vision-only        # just scene/purge classification
pixel-purge classify --faces-only         # just face clustering
pixel-purge classify --stats              # classification statistics

pixel-purge review                        # approve/reject flagged items (TUI)

pixel-purge cleanup                        # export a deletion manifest (safe default)
pixel-purge cleanup --strategy stage       # copy keepers to staging + restore metadata
pixel-purge export -o manifest.csv         # full manifest CSV
```

Cleanup never deletes via the API (Google removed those scopes and never
supported media deletion). The default `cleanup` writes a reviewable deletion
manifest; `clean-slate` is a gated last resort requiring `--i-have-a-backup` and
`--confirm "DELETE MY LIBRARY"`. Upload/browser paths need the `[gphoto]`/`[browser]`
extras.

Duplicates and purge candidates are **flagged for review, never auto-deleted** —
the human decision (approve/reject) is recorded first, and even then deletion is
manual or via the experimental browser driver.

## Development

```bash
pip install -e '.[dev]'
pytest
```
