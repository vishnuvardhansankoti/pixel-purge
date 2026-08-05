# Pixel Purge

**Fully local-first Google Photos cleanup, deduplication, and classification engine.**

Years of Google Photos accumulate duplicates, screenshots, receipts, and blurry junk.
Pixel Purge works from a **Google Takeout export on your own machine** to find duplicates,
classify every photo (keep / trip / purge), cluster faces, and produce a reviewed deletion
plan — with **no cloud hosting, no third-party data store, and nothing auto-deleted**. The
only network call is to Google Photos itself (with your own credentials) if you choose to
re-upload a cleaned library.

See [`docs/pixel-purge-prd.md`](docs/pixel-purge-prd.md) for the full design.

---

## Table of contents

- [How it works](#how-it-works)
- [Requirements](#requirements)
- [1. Get your photos with Google Takeout](#1-get-your-photos-with-google-takeout)
- [2. Install](#2-install)
- [3. Quick start](#3-quick-start)
- [End-to-end workflow](#end-to-end-workflow)
- [Command reference](#command-reference)
- [Configuration](#configuration)
- [Classification buckets](#classification-buckets)
- [Safety model](#safety-model)
- [Monthly upkeep (delta)](#monthly-upkeep-delta)
- [Development](#development)

---

## How it works

A single local **SQLite manifest** (`~/.pixel-purge/manifest.db`) is the source of truth.
The pipeline fills it in stages:

```
Takeout export ─▶ ingest ─▶ dedup ─▶ classify ─▶ review ─▶ cleanup
                  (A)        (B)       (C)         (TUI/UI)   (D)
```

- **Ingest (A)** — discover media, merge Takeout sidecar JSON (GPS, timestamps) + EXIF.
- **Dedup (B)** — 3-tier: exact SHA-256 → spatiotemporal bucketing → perceptual-hash near-dup.
- **Classify (C)** — local CLIP zero-shot into 4 buckets + InsightFace face clustering.
- **Review** — approve/reject flagged items in a terminal UI or the local web dashboard.
- **Cleanup (D)** — export a deletion manifest (safe default), stage keepers, or re-upload.

Everything runs on-device. Duplicates and purge candidates are **flagged, never auto-deleted**.

---

## Requirements

| Requirement | Notes |
|---|---|
| **Python 3.11+** | |
| **macOS (Apple Silicon)** | Primary target; Linux/Windows work for the core CLI |
| **ffmpeg** (optional) | Video keyframes — `brew install ffmpeg`. Without it, videos still get exact-hash dedup |
| **exiftool** (optional) | Metadata restore for HEIC/PNG/video during staging — `brew install exiftool` |

ML features (`classify`) download model weights on first run (~350 MB CLIP + ~300 MB InsightFace).

---

## 1. Get your photos with Google Takeout

Pixel Purge never touches your live Google account to read photos — you export them once via
[Google Takeout](https://takeout.google.com).

1. Go to **[takeout.google.com](https://takeout.google.com)** (sign in with your Google account).
2. Click **Deselect all**, then scroll down and check **only "Google Photos"**.
3. *(Optional)* Click **All photo albums included** to pick specific albums/years.
4. Scroll to the bottom, click **Next step**.
5. Under **Transfer**, choose **Send download link via email**.
6. Set:
   - **Frequency:** *Export once*
   - **File type:** **`.tgz`** (recommended — Pixel Purge auto-extracts it; `.zip` also works if you unzip first)
   - **File size:** `50 GB` (fewer files to juggle; Google still splits very large libraries into `takeout-*.tgz` parts)
7. Click **Create export**. Google emails a download link when it's ready (minutes to hours,
   depending on library size).
8. Download all parts into one folder, e.g. `~/Downloads/pixel-purge-takeout/`. You'll have
   files like `takeout-20260805T000000Z-001.tgz`, `-002.tgz`, …

You can point `pixel-purge ingest` at:
- the **folder containing the `.tgz` parts** (it extracts them all), or
- a single **`.tgz` file**, or
- an already-**extracted directory** (the `Takeout/Google Photos/…` tree).

> **Tip:** Keep the original Takeout download until you've verified results — it's your backup.

---

## 2. Install

```bash
git clone https://github.com/vishnuvardhansankoti/pixel-purge.git
cd pixel-purge
python -m venv .venv && source .venv/bin/activate

pip install -e .              # core: ingest + dedup + cleanup
```

Optional extras (add the ones you need):

```bash
pip install -e '.[heic]'      # HEIC / HEIF (iPhone) decoding
pip install -e '.[raw]'       # camera RAW decoding (CR2/NEF/ARW/DNG)
pip install -e '.[vision]'    # CLIP classification (torch, open_clip)
pip install -e '.[faces]'     # InsightFace + DBSCAN (onnxruntime, scikit-learn)
pip install -e '.[dashboard]' # local web dashboard (FastAPI, uvicorn)
pip install -e '.[gphoto]'    # Google Photos re-upload (OAuth)
```

Or grab several at once: `pip install -e '.[heic,vision,faces,dashboard]'`.

The core CLI (ingest, dedup, cleanup export) works with **no extras**.

---

## 3. Quick start

The minimum path from a Takeout folder to a reviewed deletion list:

```bash
source .venv/bin/activate

# Point at your downloaded Takeout folder (auto-extracts .tgz parts)
pixel-purge ingest ~/Downloads/pixel-purge-takeout/

# Find duplicates (flags only — nothing is deleted)
pixel-purge dedup

# Classify + cluster faces (needs [vision] and [faces] extras)
pixel-purge classify

# Review visually in the browser (needs [dashboard] extra)
pixel-purge dashboard          # opens http://localhost:8787

# Export the reviewed deletion plan to CSV
pixel-purge cleanup            # writes deletions.csv
```

Then delete the listed items in Google Photos yourself, or use the gated `clean-slate`
re-upload flow (below).

---

## End-to-end workflow

### Step 1 — Ingest

```bash
pixel-purge ingest ~/Downloads/pixel-purge-takeout/
pixel-purge ingest ~/Downloads/pixel-purge-takeout/ --dry-run   # preview, write nothing
```

Discovers all media, merges each file's Takeout sidecar JSON (GPS, capture time, description)
and falls back to embedded EXIF. Videos get multiple keyframes + duration extracted (needs
ffmpeg). Creates `~/.pixel-purge/manifest.db` if absent. **Resumable** — re-running skips files
already ingested (keyed on full path), so an interrupted run just continues.

### Step 2 — Deduplicate

```bash
pixel-purge dedup                 # all 3 tiers
pixel-purge dedup --tier 1        # exact-hash only
pixel-purge dedup --stats         # show duplicate counts + reclaimable space
```

- **Tier 1** exact SHA-256 duplicates.
- **Tier 2** groups items by GPS + time proximity (so Tier 3 only compares plausibly-related shots).
- **Tier 3** perceptual-hash near-duplicates within each group. Screenshots/documents get a
  stricter, near-exact threshold so distinct text images aren't wrongly merged; videos are
  matched across multiple frames with a duration gate.

Duplicates are marked `is_duplicate`, with a keeper auto-chosen (richest metadata → earliest
capture → largest file). Nothing is deleted — the keeper choice is reviewable.

### Step 3 — Classify + cluster faces

```bash
pixel-purge classify              # CLIP vision + face clustering
pixel-purge classify --vision-only
pixel-purge classify --faces-only
pixel-purge classify --stats
```

CLIP scores each photo into one of four [buckets](#classification-buckets); blur and text-density
heuristics refine screenshots/documents/blurry shots into `ADHOC_PURGE`. Faces are embedded with
InsightFace and grouped into people with DBSCAN. Resumable per stage.

### Step 4 — Review

**Terminal:**
```bash
pixel-purge review                # batch approve/reject flagged items
```

**Browser (recommended):**
```bash
pixel-purge dashboard             # http://localhost:8787
```

The dashboard has three tabs:
- **Review** — grid of duplicates + purge candidates with thumbnails, bucket, confidence, and
  Keep/Delete buttons (bucket filters included).
- **Dedup** — side-by-side duplicate clusters with the keeper highlighted.
- **Faces** — person clusters; name them, merge split clusters.

Your Keep/Delete decisions are written back to the manifest immediately.

### Step 5 — Cleanup

```bash
pixel-purge cleanup                       # export deletions.csv (safe default)
pixel-purge cleanup --strategy stage      # copy keepers to a staging dir + restore metadata
pixel-purge export -o manifest.csv        # full manifest as CSV
```

`cleanup` (default) writes a reviewable **deletion manifest** of everything you marked `DELETE`.
Because Google's API can no longer delete library media, you then either:

- **Delete manually** in Google Photos using the manifest as your checklist, **or**
- **Clean-slate re-upload** (advanced, gated — see [Safety model](#safety-model)):

```bash
# 1. Stage the keepers locally with metadata restored
pixel-purge cleanup --strategy stage --staging-dir ~/pixel-purge-staging

# 2. After you manually wipe your library + empty trash, re-upload the clean set
pixel-purge cleanup --strategy clean-slate \
    --i-have-a-backup --confirm "DELETE MY LIBRARY"
pixel-purge cleanup --strategy upload     # pushes staged files (needs [gphoto])
```

---

## Command reference

| Command | Purpose |
|---|---|
| `pixel-purge init` | Create the manifest DB (`ingest` does this automatically too) |
| `pixel-purge ingest <path>` | Ingest a Takeout `.tgz`/folder; `--dry-run`, `--no-resume`, `--no-keyframes` |
| `pixel-purge dedup` | 3-tier deduplication; `--tier N`, `--stats`, threshold flags |
| `pixel-purge classify` | CLIP vision + face clustering; `--vision-only`, `--faces-only`, `--stats` |
| `pixel-purge review` | Terminal batch approve/reject |
| `pixel-purge dashboard` | Local web UI; `--port`, `--host`, `--no-open` |
| `pixel-purge cleanup` | Export deletions / stage / clean-slate / upload / browser-auto |
| `pixel-purge export -o file.csv` | Full manifest CSV; `--deletions-only` |
| `pixel-purge delta <path>` | Monthly incremental classify; `--dry-run`, `--stats` |
| `pixel-purge schedule <path>` | Install a monthly launchd agent (macOS) |
| `pixel-purge eval labeled.csv` | Measure classifier accuracy vs a labeled set |
| `pixel-purge version` | Print version |

Run any command with `--help` for its full options.

---

## Configuration

Optional TOML at `~/.pixel-purge/config.toml`. All keys have sensible defaults; set
`[home_base]` if you want accurate trip detection.

```toml
[general]
db_path = "~/.pixel-purge/manifest.db"
log_level = "INFO"

[home_base]                 # used by delta to flag far-from-home photos as TRIP
latitude = 0.0
longitude = 0.0

[dedup]
gps_radius_meters = 100
time_window_minutes = 30
hamming_threshold = 8       # near-dup match distance for photos
text_hamming_threshold = 2  # stricter, near-exact, for screenshots/documents

[delta]
trip_distance_miles = 50
model = "ViT-B-32"          # local CLIP checkpoint
device = "auto"             # auto | mps | cpu
notify = true               # macOS notification when review candidates are ready
```

---

## Classification buckets

Both `classify` and `delta` use one taxonomy:

| Bucket | Meaning |
|---|---|
| `ADHOC_PURGE` | Screenshots, receipts, documents, memes, QR codes, blurry — deletion candidates |
| `TRIP` | Travel, landmarks, scenery (GPS-verified as far from home when coordinates exist) |
| `FAMILY_KEEP` | People, family/children, pets, gatherings, food, celebrations |
| `OTHER` | Low-confidence / miscellaneous — always routed to review, never auto-purged |

Measure accuracy on your own labeled data:

```bash
# labeled.csv: columns  path,bucket
pixel-purge eval labeled.csv        # confusion matrix + precision/recall/F1 + accuracy vs --target
```

---

## Safety model

- **Nothing is auto-deleted.** Dedup and classify only *flag*; you approve/reject, and the
  decision is recorded in the manifest.
- **The API never deletes** your library (Google removed those scopes and never supported media
  deletion). The default output is a reviewable CSV you act on manually.
- **Clean-slate is a gated last resort.** Wiping and re-uploading is irreversible (album shares,
  Memories, and Google's own grouping are lost), so it requires **both**
  `--i-have-a-backup` **and** `--confirm "DELETE MY LIBRARY"`. Keep your original Takeout as the backup.
- **Browser deletion is experimental**, opt-in (`--strategy browser-auto`, `[browser]` extra),
  and targets items by their stored Google Photos URL — verify results afterward.
- **Local-only.** The manifest and dashboard never leave your machine; the dashboard binds to
  `localhost` (the OS user is the security boundary).

---

## Monthly upkeep (delta)

Keep the library clean after the big one-time purge. Each month, export a small incremental
Takeout (last month of photos) and run:

```bash
pixel-purge delta ~/Downloads/takeout-2026-09.tgz    # ingest + dedup + classify new items
pixel-purge delta --stats                            # per-bucket counts + watermark
```

`delta` is idempotent — a stored watermark means re-running the same export classifies nothing
new. Automate it on macOS:

```bash
pixel-purge schedule ~/Downloads/pixel-purge-takeout    # writes a launchd plist
# then: launchctl load ~/Library/LaunchAgents/com.pixelpurge.delta.plist
```

---

## Development

```bash
pip install -e '.[dev]'
pytest                      # full suite runs without the heavy ML/cloud extras
```

The test suite mocks the model layers, so `pytest` is fast and needs no torch/onnxruntime.
