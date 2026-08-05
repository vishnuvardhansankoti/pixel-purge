You are an expert Principal Systems Architect and Technical Product Manager. Your task is to generate a comprehensive, engineering-ready Product Requirements Document (PRD) for a system titled: "Google Photos Cleanup, Classification, and Deduplication Agent System".

Use the context, technical constraints, and architectural decisions outlined below to author the PRD.

---

### CONTEXT & SYSTEM OVERVIEW
The target system addresses the challenge of managing large (50GB+, 20,000+ items) Google Photos libraries where native API limits restrict bulk deletions, automated tagging, and deep deduplication. 

The architecture is split into two primary components:
1. **Local Bulk Engine (Batch Phase):** A local-first CLI/Python pipeline processing a full Google Takeout export for initial deduplication, metadata restoration, visual labeling, and face clustering.
2. **Monthly Serverless Classifier (Delta Sync Phase):** An automated cloud worker running on a monthly schedule to classify new media (past 30 days) into Family, Trip, or Adhoc Junk buckets and stage candidates for review.

---

### CRITICAL TECHNICAL CONSTRAINTS & PITFALLS TO ADDRESS
1. **API Limitations:** Google Photos REST API does not allow third-party apps to execute bulk deletions or empty trash bins.
2. **JSON Sidecar Split:** Google Takeout exports strip or isolate EXIF metadata (GPS `geoData`, creation timestamps) into companion `.json` files.
3. **Server-Side Duplicate Rejection:** Google Photos uses server-side bit-hash deduplication. Uploading identical photo copies back to a `TO_DELETE` album is silently blocked by Google's servers.
4. **Processing Efficiency:** Direct $O(N^2)$ visual comparison across 20,000 photos (~200M operations) is infeasible. Comparison must be filtered hierarchically.

---

### REQUIRED PRD STRUCTURE

Generate the PRD using the following structured sections:

#### 1. Executive Summary & Goals
* Core vision, target user persona, and primary success metrics (e.g., storage reclaimed, manual cleanup time reduced by 90%).

#### 2. System Architecture & Component Specifications
Detail the specifications and step-by-step logic for these 5 core modules:
* **Module A: Ingestion & Metadata Merger:**
  * Parsing Google Takeout archives.
  * Merging `.json` / `.supplemental-metadata.json` sidecar fields (`latitude`, `longitude`, `photoTakenTime`, `description`) back into media records.
* **Module B: Hierarchical Deduplication Pipeline:**
  * Tier 1: $O(N)$ Binary hashing (MD5/SHA-256) for exact bit-level duplicates.
  * Tier 2: Spatiotemporal partitioning (grouping by GPS distance and time windows).
  * Tier 3: Visual Perceptual Hashing (`pHash` via `imagehash`) with Hamming distance thresholds ($\le 10$) executed *only* within local spatiotemporal buckets.
* **Module C: Local AI Vision & Unsupervised Face Clustering:**
  * Object & Scene Tagging: Local Vision Transformer (`Salesforce/blip-image-captioning-base`) to identify documents, receipts, screenshots, and blur.
  * Human Face Clustering: `dlib` 128-dimensional face embedding extraction paired with `scikit-learn` `DBSCAN` density clustering (`eps=0.55`) to assign `person_cluster_id` across solo and group photos without manual labeling.
* **Module D: Cloud Cleanup Execution Engine:**
  * Specify two supported deletion strategies:
    1. *Clean Slate Strategy:* Local folder cleanup -> Cloud account wipe -> Pristine re-upload.
    2. *Browser Automation Strategy:* Macro script (Playwright/Selenium) reading the generated manifest to execute search queries (`photos.google.com/search/[filename]`) and automate deletion.
* **Module E: Monthly Serverless Delta Sync Classifier:**
  * Trigger: GCP Cloud Scheduler / AWS EventBridge cron (1st of every month).
  * Runtime: GCP Cloud Run container (Python).
  * Ingestion: Google Photos API `mediaItems:search` for the last 30 days delta (~1-2GB).
  * Multimodal Inference: Gemini 1.5 Flash API or lightweight vision LLM classifier with 3-bucket output logic:
    * *Bucket 1 (ADHOC_PURGE):* Screenshots, receipts, OCR text > 30%, blurry images.
    * *Bucket 2 (TRIP):* GPS location > 50 miles (80 km) from home base or high spatial/temporal cluster density.
    * *Bucket 3 (FAMILY_KEEP):* Recognized facial cluster embeddings or human social activity.
  * Staging & Alerting: Auto-creation of `Review_For_Deletion_YYYY_MM` Google Photos album + Webhook payload sent to Discord/Telegram with direct action links.

#### 3. Data Schemas & Manifest Specifications
Define the formal database/CSV schema for the central manifest (`photo_agent_manifest.csv`):
* Field names, data types, nullability, and description for fields including: `filename`, `local_path`, `cloud_timestamp`, `latitude`, `longitude`, `phash_cluster_id`, `ai_generated_label`, `person_cluster_id`, and `classification_bucket`.

#### 4. Non-Functional Requirements
* Performance & Execution Time targets.
* Data Privacy & Local-First security guarantees.
* Cloud Infrastructure Cost Budgets (aiming for $0.00/mo within GCP/AWS Free Tiers).
* Error handling (corrupted EXIF headers, missing JSON files, API rate-limit exponential backoff).

#### 5. Implementation Roadmap & Functional Milestones
* Phase 1: Local Takeout Parser & Deduplicator CLI.
* Phase 2: Local Vision & Face Clustering Agent.
* Phase 3: Web Automation / Cleanup Execution Module.
* Phase 4: Serverless Monthly Cloud Run Worker & Webhooks.

---
Tone: Professional, highly structured, technical, and complete. Avoid placeholders or truncated code snippets—provide exact specifications and pseudo-code where applicable.