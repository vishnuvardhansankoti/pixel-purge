# Pixel Purge PRD — Architecture Review

**Reviewer role:** Principal AI Solutions Architect
**Reviewed doc:** [`pixel-purge-prd.md`](./pixel-purge-prd.md) — reviewed at v1.0, updated below for **v1.1**
**Review date:** 2026-08-04 · **Last updated:** 2026-08-04 (post v1.1 all-local pivot)

**Verdict:** Strong engineering craft. The v1.0 draft had two load-bearing external assumptions that
were broken as of 2026 (Google Photos library-read API; Gemini 1.5 Flash) and one architectural seam
that could not work as drawn (a hosted PWA showing local files). **v1.1 resolves all three by going
fully local** — no cloud runtime, scheduler, data store, hosting, or managed model. The remaining
open findings are all in the batch pipeline (Modules A–D) and are correctness/quality items, not
blockers.

Severity legend: 🔴 blocks the design · 🟠 high-risk · 🟡 correctness/quality.
Status legend: ✅ resolved in v1.1 · ➖ moot in v1.1 (component removed) · ⚠️ open (still to fix).

---

## 1. Findings (ranked)

### 🔴 Critical

| ID | Finding | Status in v1.1 |
|---|---|---|
| **C1** | **Google Photos Library API can no longer read the user's library.** Since 2025-04-01 the `photoslibrary`, `photoslibrary.readonly`, `photoslibrary.sharing` scopes were removed; `mediaItems.search`/`list`/`batchGet` now return **only app-created media**. "Fetch last 30 days of everything" is impossible; full-library selection needs the interactive Photos Picker API. | ✅ **Resolved.** §2.7 delta is now fully local (incremental Takeout / Picker export → local pipeline). No library-wide API read. Upload/delete via `appendonly` still valid. |
| **C2** | **"Clean Slate" (wipe entire cloud library → re-upload) is irreversible with no rollback.** A single keeper/pHash error silently destroys originals; album shares, Memories, shared links, Google face groups, comments are permanently lost; re-upload counts against storage quota. | ⚠️ **Open.** Module D §2.6 still lists clean-slate as primary. Demote to last-resort behind a verified backup + typed confirm; make targeted deletion primary. |
| **C3** | **Hosted PWA cannot display local-library thumbnails** — a Firebase-hosted browser app cannot load `file:///…`, so the Dedup/Face views would render broken tiles. | ✅ **Resolved.** §2.8 is a single **local dashboard** (FastAPI on `localhost`) streaming thumbnails from disk via `/thumb/{id}`. |
| **C4** | **Gemini 1.5 Flash is retired** (unavailable to new projects since 2025-04-29; current gen is Gemini 3.5/3.6 Flash); code also used the deprecated `google.generativeai` surface. | ➖ **Moot.** Delta classification is now **local CLIP zero-shot**; no hosted LLM in the system. |

### 🟠 High

| ID | Finding | Status in v1.1 |
|---|---|---|
| **H1** | pHash Hamming ≤10 on screenshots/docs causes **false-positive deletions** — distinct receipts/boarding-passes/chats routinely sit within 10 bits; under clean-slate = silent data loss, blowing the <0.1% metric. | ⚠️ **Open.** Tier 3 §2.4 still uses threshold 10. Tighten to ≤6; disable near-dup merge for `SCREENSHOT/DOCUMENT/RECEIPT`; never auto-delete on a pHash-only match. |
| **H2** | **BLIP caption→keyword won't reach ≥90%**, and batch vs delta used **incompatible taxonomies** with no mapping; no labeled eval set defined. | ◐ **Partially resolved.** v1.1 defines **one unified taxonomy** (§2.7) and adopts **CLIP zero-shot** for delta. Still open: migrate batch Module C (§2.5) off BLIP+keyword to CLIP, and **add the labeled eval set** referenced by P2.2/P4.2. |
| **H3** | **Tier-2 GPS clustering is spatially incorrect**: lexicographic `(lat,lon)` sort doesn't preserve locality; greedy distance uses only the anchor (splits walks, merges neighbors); fixed ±30-min windows split bursts on boundaries. | ⚠️ **Open.** §2.4 unchanged. Use **geohash** bucketing or haversine-metric DBSCAN; **gap-based sessionization** for time-only. |
| **H4** | `face_recognition`/dlib has **no MPS backend + painful M1 build**; HOG misses profile/small faces; **`min_samples=3` silently drops anyone in ≤2 photos**. | ⚠️ **Open.** §2.5 unchanged. Use **InsightFace (buffalo_l, ONNX + CoreML EP)**; `min_samples=2`; surface singletons instead of discarding. |
| **H5** | **Monthly idempotency drifts**: rolling "last 30 days" vs calendar-month cron → boundary gaps/overlaps; overlaps re-stage the same photos each run. | ✅ **Resolved.** §2.7 persists `last_delta_watermark` and classifies only items newer than it. |

### 🟡 Medium

| ID | Finding | Status in v1.1 |
|---|---|---|
| **M1** | **Resume keys on filename, not path** — Takeout reuses basenames; `local_path` is UNIQUE, so resume wrongly skips distinct files. | ⚠️ **Open.** `ingest()` §2.3 unchanged. Key resume on full path / path hash. |
| **M2** | **EXIF restore fails for non-JPEG** — `piexif` is JPEG/TIFF only; PNG/HEIC/video get no GPS/timestamp → wrong dates after re-upload. | ⚠️ **Open.** `restore_exif` §2.6 unchanged. Format-aware strategy (`exiftool`, `pillow-heif`, ffmpeg). |
| **M3** | **HEIC/RAW can't be opened by PIL/imagehash** without `pillow-heif`/`rawpy`; large share of an iPhone library errors. | ◐ **Partial.** The §2.8 `/thumb` code now flags the `pillow-heif/rawpy` need, but §2.3 ingestion + Module C decoding still don't declare the deps. Add them explicitly. |
| **M4** | Exact-hash keeper heuristics contradictory ("highest-resolution" is moot for byte-identical dupes); `select_keeper(ids)` needs item data, not just IDs. | ⚠️ **Open.** Tier 1 §2.4 unchanged. Metadata/path tiebreak only; make DB contract explicit. |
| **M5** | **Playwright deletion assumes filename search works** — Google Photos doesn't reliably index original filenames. | ⚠️ **Open.** Module D Strategy 2 §2.6 unchanged. Drive from grid + date nav, not filename search. |
| **M6** | **Owner-email config drift** across Firestore rules / `ALLOWED_EMAIL` / `config.toml`. | ➖ **Moot.** No Firestore/hosted auth in v1.1; the local dashboard has no owner-email concept. |
| **M7** | **`$0.00` required Blaze billing**; AI Studio free tier trains on inputs — contradicted §4.2. | ➖ **Moot.** §4.3 is now "no hosted infrastructure, no billing account required"; no cloud inference. |
| **M8** | Storing `base_url` in Firestore is dead weight (expires ~60 min). | ➖ **Moot.** No Firestore; the local dashboard renders thumbnails from disk. |
| **M9** | Single t=1s keyframe is fragile for video near-dup (misses re-encodes; false-matches similar openings). | ⚠️ **Open.** §2.3 unchanged. Multi-frame hash + duration/size gating. |

---

## 2. What the v1.1 all-local pivot changed

The design decision — **fully local, no cloud hosting of any kind** — was applied across the PRD:

- **§1.1 / §2.1 / §2.2:** vision, architecture diagram, and tech stack now describe a local-only
  system; the only network egress is Module D → Google Photos (upload / targeted delete).
- **§2.7 Module E:** rewritten as a **local monthly delta** — CLIP zero-shot classifier, unified
  4-bucket taxonomy (`ADHOC_PURGE`/`TRIP`/`FAMILY_KEEP`/`OTHER`), watermark idempotency, `launchd`
  scheduling. Serverless / Gemini / Firestore paths removed.
- **§2.8 Module F:** rewritten as a **single local dashboard** (FastAPI + SvelteKit build on
  `localhost`) with an on-disk `/thumb/{id}` endpoint serving Delta Review, Dedup, and Face views.
  Firebase Hosting/Auth/Firestore and the `sync-dashboard` upload removed.
- **§3.2 config, §4.2 privacy, §4.3 cost, §5 roadmap + directory tree:** all cloud config keys,
  the GCP/Firebase cost table, the Cloud Run/Firestore roadmap phases, and the
  Dockerfile/firebase/deploy files were replaced with their local equivalents. `OTHER` added to the
  `classification_bucket` CHECK constraint and CSV schema.

This resolves **C1, C3, C4, H5** and moots **M6, M7, M8**; **H2** and **M3** are partially resolved.

---

## 3. Remaining work (prioritized) — all in the batch pipeline

The local pivot did not touch Modules A–D. These open findings should gate Phase 1–3 sign-off:

1. **Silent-data-loss bugs first (Phase 1):** resume-by-path **[M1]** and format-aware decode/EXIF
   **[M2, M3]** — they corrupt the manifest before anything downstream runs.
2. **Deletion safety:** make targeted deletion primary and gate clean-slate behind a verified backup
   **[C2]**; never auto-delete on a pHash-only match, and tighten the threshold **[H1]**.
3. **Dedup correctness:** geohash/DBSCAN spatial bucketing + gap-based sessionization **[H3]**;
   fix keeper heuristics **[M4]**; multi-frame video hashing **[M9]**.
4. **Vision/faces:** migrate batch Module C to **CLIP** for taxonomy parity with delta and add the
   **labeled eval set** **[H2]**; swap faces to **InsightFace**, `min_samples=2` **[H4]**.
5. **Cleanup robustness:** if Playwright deletion is kept, drive it from the grid, not filename
   search **[M5]**.

---

## Sources

- [Google Photos — Updates to the APIs](https://developers.google.com/photos/support/updates)
- [Google Photos — mediaItems.search reference](https://developers.google.com/photos/library/reference/rest/v1/mediaItems/search)
- [Google Photos — Access app-created media items](https://developers.google.com/photos/library/guides/access-media-items)
- [Gemini API — deprecations](https://ai.google.dev/gemini-api/docs/deprecations)
- [Gemini API — changelog](https://ai.google.dev/gemini-api/docs/changelog)
