---
noteId: "1e1e7b602fbb11f1b60c85121fd90d06"
tags: []

---

# Data Inventory — `data-gold-final/`

This document inventories all data in the `data-gold-final/` directory, the primary data store for the Smartflat project (SDS2 study). Each section maps folders to the pipeline stage that created them, labels their production status, and describes their contents.

**Data root resolution:** The pipeline locates this directory via `get_data_root()` (in `smartflat/utils/utils_paths.py`), which checks `SMARTFLAT_DATA_ROOT` env var first, then falls back to host-specific paths.

## Top-Level Structure

```
data-gold-final/
├── cuisine/              # Raw recordings + extracted features (198 participants)
├── dataframes/           # Metadata, annotations, clinical data, quality control
├── experiments/          # Clustering labels, CPD results, distance matrices
├── experiments_speech/   # Speech diarization analysis outputs
├── outputs/              # Symbolization state, visualizations, CPD deployment outputs
└── thumbnails/           # Video thumbnail caches for annotation tools
```

| Directory | Status | Description |
|-----------|--------|-------------|
| `cuisine/` | PRODUCTION | Participant folders with raw recordings and extracted features |
| `dataframes/` | PRODUCTION | Metadata DataFrames, annotations, clinical scores |
| `experiments/` | MIXED | Experiment results (clustering labels, distance matrices) |
| `experiments_speech/` | EXPLORATORY | Speech diarization visualizations (not in main pipeline) |
| `outputs/` | MIXED | Symbolization pipeline state and visualizations |
| `thumbnails/` | SUPPORT | Thumbnail caches for pigeon annotation tool |

---

## 1. Raw Recordings and Features — `cuisine/`

**Status:** PRODUCTION  
**Created by:** Manual data collection (raw) + feature extraction modules

### 1.1 Participant Folder Structure

198 participant folders following the naming convention:

```
G{group_id}_{participant_type}{participant_id}_{trigram}_{date}/
```

- `G{id}`: Sequential group number (G1–G115, with gaps)
- `{type}`: `C` = control, `P` = patient
- `{trigram}`: 3-letter first-name + 3-letter last-name code (e.g., `BAUVin`)
- `{date}`: Recording date as `ddmmyyyy`

Example: `G100_P86_BAUVin_25112022/`

### 1.2 Modality Subfolders

Each participant folder contains standardized modality subfolders (created by `features/consolidation/main_init_folder_structure.py`):

| Subfolder | Contents | Status |
|-----------|----------|--------|
| `Tobii/` | Eye-tracking video (.mp4), audio (.wav), extracted features | PRODUCTION — primary modality |
| `GoPro1/` | Fixed camera 1 video + hand landmarks | PRODUCTION |
| `GoPro2/` | Fixed camera 2 video | PRODUCTION (raw only) |
| `GoPro3/` | Fixed camera 3 video | PRODUCTION (raw only) |
| `Annotation/` | BORIS/Vidat annotation exports | PRODUCTION (when populated) |
| `Audacity/` | Speech segmentation exports | LEGACY (mostly empty) |

### 1.3 Feature Files Inside Modality Folders

Feature extraction modules write outputs directly into modality folders. The primary modality is `Tobii/`:

| Feature | File Pattern | Size (typical) | Module |
|---------|-------------|-----------------|--------|
| VideoMAE-v2 (D=1408) | `video_representations_VideoMAEv2_{video}.npy` | ~50–60 MB | `features/video/main.py` |
| Hand landmarks | `hand_landmarks_mediapipe_{video}.json` | ~70–240 MB | `features/hands/main.py` |
| Skeleton pose | `skeleton_landmarks_{video}.json` | variable | `features/skeleton/main.py` |
| Speech recognition | `speech_recognition_diarization_whisperx_{video}.json` | variable | `features/audio/main.py` |
| Speech embeddings | `speech_representations_multilingual_{video}.npy` | variable | `features/audio/main.py` |
| Hand tracking | `tracking_hand_landmarks_{video}.json` | variable | `features/hands_processing/main.py` |

**Flag files** track extraction completion: `.{video}_{feature_type}_flag.txt` (hidden files, 7 bytes each). The code checks these before re-running extraction.

**Video names** are Tobii-assigned base64 IDs (e.g., `LY4N1_iUVPJHz1jIw-r79Q==`) or participant-folder names for GoPro sources.

### 1.4 Legacy: `cuisine/experiments/`

**Status:** LEGACY  
Contains `clustering-deployment/` and `clustering-deployment-kmeans-all/`. These are misplaced experiment folders that should be under the top-level `experiments/` directory. Likely created by an early version of the clustering pipeline before the output path conventions were standardized.

---

## 2. Metadata and Annotations — `dataframes/`

**Status:** PRODUCTION  
**Created by:** `datasets/build.py`, annotation tools, clinical data collection

### 2.1 Core Metadata Files

| File | Description | Created By |
|------|-------------|------------|
| `gold_dataset_df_backup.csv` | Main dataset metadata (1.6 MB, all participants) | `datasets/build.py` |
| `egr-sjp49-mbp.local_dataset_df.csv` | Host-specific dataset snapshot | `datasets/build.py` |
| `df_cuisine_tobii_with_segmentation.csv` | Cuisine-Tobii subset with segmentation info | Symbolization pipeline |
| `tableau_annotation_cuisine_Smartflat.csv/.npy` | Consolidated BORIS annotation table | `annotation_smartflat.py` |
| `target_embedding.npy` | Pre-computed target embedding array (3.6 MB) | Symbolization pipeline |

### 2.2 Annotations — `annotations/`

**Status:** PRODUCTION

| Path | Description |
|------|-------------|
| `annotations_all.csv` | Consolidated BORIS annotations for all participants (8.5 MB) |
| `annotations_all_29102024.csv` | Backup from Oct 2024 |
| `annotation_codebook_boris.csv` | BORIS codebook defining A–J activity categories |
| `missing_annotation_mapping_29102024.csv` | Mapping for missing annotation entries |

#### Pigeon Annotations — `annotations/pigeon-annotations/`

**Status:** PRODUCTION  
**Created by:** Pigeon annotation tool during recursive prototyping

Human annotators (samperochon, theoperochon) qualified cluster prototypes across 8 rounds. Structure:

```
pigeon-annotations/
├── symbolization-gold/
│   ├── {experiment_id}/           # e.g., faissc_refinement_symbolization
│   │   └── samperochon/
│   │       └── round_{N}_prototypes_K_100.csv    # N = 1..8
│   ├── cosine_iteration_1/
│   ├── euclidean_iteration_1/
│   ├── faissc_iteration_1/
│   ├── faissc_inference_symbolization/
│   ├── faissc_merge_inference_symbolization/
│   ├── faissc_refinement_symbolization/          # Main production annotations
│   └── faisse_iteration_1/
├── init_clustering_cosine/
├── init_clustering_euclidean/
├── init_clustering_faissc/
├── init_clustering_faisse/
├── clustering-deployment-kmeans-all/
├── clustering-deployment-kmeans-task-r1/
├── clustering-deployment-kmeans-task-r3/
└── clustering-deployment-kmeans-task-r4/
```

Each annotation CSV contains cluster prototype qualifications: `task-definitive`, `exo-definitive`, `task-ambiguous`, `exo-ambiguous`, `Noise`.

`pigeon-annotations-010132025/` is a dated backup of the annotations directory.

### 2.3 Clinical Data — `clinical/`

**Status:** PRODUCTION

| File | Description |
|------|-------------|
| `clinical_data_mupt.csv` | MUPT clinical scores |
| `merged-clinical-data-mupt.csv` | Merged MUPT data |
| `clinical_metadata.csv` | Clinical metadata |
| `bilan-cliniques-sds2-17102024.csv` | SDS2 clinical assessments (Oct 2024) |
| `bilan-data-sds2-17102024.csv` | SDS2 assessment data |
| `codebook-data-sds2-17102024.csv` | SDS2 variable codebook |
| `roue-emotions-sds2-17102024.csv` | Emotion wheel results |
| `data-pre-sds2-df2_man1_an-24052024.csv` | SDS2 pre-study data |
| `percy_participant_id_to_rename.csv` | ID mapping between Percy hospital and study IDs |
| `metadata_raw_completed.csv` | Complete raw metadata (1.3 MB) |
| `MoCA.pdf` | Montreal Cognitive Assessment reference |
| `Scores_cliniques.csv` | Summary clinical scores |
| `backup/` | Backup subfolder |

### 2.4 Quality Control — `quality-control/`

**Status:** PRODUCTION  
**Created by:** `datasets/quality_control.py`

Iterative visual inspection results across 5 CSV files:

| File | Description |
|------|-------------|
| `results_visual_inspection.csv` | Initial QC pass |
| `results_visual_inspection_2.csv` | Second pass |
| `results_visual_inspection_all.csv` | Combined results |
| `results_visual_inspection_final.csv` | Final consolidated QC |
| `results_visual_inspection_2111124_last.csv` | Latest QC results (568 KB) |

### 2.5 Persistent Metadata — `persistent_metadata/`

**Status:** SUPPORT (audit trail)

Contains 60+ files including:
- **Timestamped backups:** `gold_dataset_df_backup_{YYYYMMDD}.csv` (20+ versions, Oct 2024 – Feb 2025)
- **Host-specific datasets:** `{hostname}_dataset_df.csv` for various machines (MacBook-Pro, MacOS-Sam-Perochon, pomme, cheetah, device-57, etc.)
- **Pre-gold metadata:** `smartflat_dataset_df_backup_{date}.csv`, `feature_extraction_dataset_df.csv`
- **Video metadata:** `smartflat_video_metadata.csv`, `smartflat_video_metadata_updated.csv`
- **Cross-correlation:** `cross_correlation.csv`, `cross_correlation_22022025.csv`
- **Raw metadata:** `metadata_raw.csv`, `metadata.csv`, `metadata_{date}_without_check.csv`

### 2.6 Other Files

| File | Status | Description |
|------|--------|-------------|
| `frozen-metrics-logs-to-merge/` | SUPPORT | Computation time logs per machine per feature type |
| `backup_lego_cake_table_post_conversion.csv` | LEGACY | Lego task data (non-cuisine) |
| `lego_cake_table_with_id.csv/.numbers` | LEGACY | Lego-cake comparison data |
| `backup_suivi longitudinal légo-Tableau 1.csv` | LEGACY | Longitudinal lego tracking |
| `comparaison_lego_gateau_liste_patients_dec22.xlsx` | LEGACY | Lego-cake patient comparison |
| `fix_DUMJEA.csv/.numbers` | LEGACY | One-off data correction for participant DUMJEA |
| `Scores_cliniques.numbers` | LEGACY | Apple Numbers version of clinical scores |

---

## 3. Experiments — `experiments/`

**Status:** MIXED  
**Created by:** Clustering engine, CPD engine, symbolization pipeline

General output path pattern (constructed in `features/symbolization/main.py`):
```
experiments/{config.experiment_name}/{config.experiment_id}/{annotator_id}/round_{N}/
```

### 3.1 Symbolization Gold — `symbolization-gold/`

**Status:** PRODUCTION  
**Created by:** Recursive prototyping pipeline (`features/symbolization/main.py`)  
**Config source:** `configs/symbolic_config_gold.py`, `configs/clustering_config.py`

Contains experiment results (cluster labels, distance matrices) for each symbolization step. 30 subdirectories organized by experiment_id:

#### Production Experiment IDs

These correspond to the 4-step production pipeline defined in `symbolic_config_gold.py`:

| Step | Config Class | experiment_id | Purpose |
|------|-------------|---------------|---------|
| Preliminary | `SymbolicSource*GoldFullConfig` | `full_cosine_iteration_1` | Initial clustering, all training samples |
| Preliminary | | `full_euclidean_iteration_1` | Euclidean variant |
| Preliminary | | `full_faissc_iteration_1` | FAISS-cosine variant |
| Preliminary | | `full_faisse_iteration_1` | FAISS-euclidean variant |
| Step 1 | `SymbolicSource*GoldConfig` | `cosine_iteration_1` | 80/20 cross-validation |
| Step 1 | | `euclidean_iteration_1` | Euclidean variant |
| Step 1 | | `faissc_iteration_1` | FAISS-cosine variant (**selected**) |
| Step 1 | | `faisse_iteration_1` | FAISS-euclidean variant |
| Step 2 | `SymbolicSourceInferenceCompleteGoldConfig` | `faissc_post_hf_symbolization` | Post-annotation inference |
| Step 3 | `SymbolicSourceInferenceRefinementGoldConfig` | `faissc_refinement_symbolization` | Refinement with manual rounds |
| Step 4 | `SymbolicSourceInferenceGoldConfig` | `faissc_inference_symbolization` | **Final inference model** |

#### Additional Production Variants (Step 4 alternatives)

| Config Class | experiment_id | Variant |
|-------------|---------------|---------|
| `SymbolicSourceMergeInferenceGoldConfig` | `faissc_merge_inference_symbolization` | Merged prototypes |
| `SymbolicSourcePrototypesTSInferenceGoldConfig` | `faissc_pts_inference_symbolization` | Prototype-based temporal segmentation |
| `SymbolicSourceReducedPrototypesTSInferenceGoldConfig` | `faissc_rpts_inference_symbolization` | Reduced prototype temporal segmentation |
| `SymbolicSourceInferenceFropFirstGoldConfig` | `faissc_inference_drop_first_symbolization` | Drop-first-round variant |

#### Internal Folder Structure

Each experiment_id folder follows:
```
{experiment_id}/
├── samperochon/
│   ├── round_0/
│   ├── round_1/
│   │   ├── config.json
│   │   ├── {participant}_labels.npy        # Cluster assignments
│   │   └── {participant}_distances.npy     # Distance to nearest centroid
│   ├── ...
│   └── round_8/
│       ├── config.json
│       ├── D_te_pc_gw_square_loss_128.npy  # Distance matrices
│       ├── D_tf_pc_wasserstein-1_*.npy     # TWE distance matrices
│       └── ...
└── theoperochon/
    └── round_1/ ... round_8/
```

**Annotator round numbers** (from `constants.py`):
- `samperochon`: rounds 1–8
- `theoperochon`: rounds 1–8
- `fusionperochon`: round 8 only

#### Init-Clustering Folders

Initialization-phase clustering results (used before recursive prototyping):

| Folder | Description |
|--------|-------------|
| `init_clustering_cosine` | L2-normalized cosine k-means |
| `init_clustering_cosine_full` | Full dataset variant |
| `init_clustering_euclidean` | Z-score euclidean k-means |
| `init_clustering_euclidean_full` | Full dataset variant |
| `init_clustering_faissc` | FAISS cosine k-means |
| `init_clustering_faissc_full` | Full dataset variant |
| `init_clustering_faisse` | FAISS euclidean k-means |
| `init_clustering_faisse_full` | Full dataset variant |

#### Non-Symbolization Folders (also under experiments/symbolization-gold/)

| Folder | Description |
|--------|-------------|
| `cosine_iteration_1_inference` | Inference-only variant of Step 1 cosine |
| `faissc_inference` | Inference without symbolization suffix |
| `faissc_post_hf` | Post-HF without symbolization suffix |
| `faissc_refinement` | Refinement without symbolization suffix |
| `faissc_merge_dinference` | Merge variant (typo: "dinference") |
| `faissc_inference_drop_first` | Drop-first without symbolization suffix |

### 3.2 Change-Point Detection

**Status:** PRODUCTION  
**Created by:** `engine/change_point_detection.py`  
**Config source:** `configs/change_points_config.py`

#### Grid Search Experiments

| Folder | Count | Config | Description |
|--------|-------|--------|-------------|
| `change-point-detection-experiment/` | 141 UUID folders | `ChangePointDetectionExperimentConfig` | Raw-space KernelCPD with varying penalty |
| `gold-change-point-detection-prototypes-experiment/` | 25 UUID folders | `ChangePointDetectionPrototypesExperimentConfig` | Prototype-space KernelCPD |

Each UUID folder contains:
- `config.json` — Hyperparameters for this run
- `{participant_id}_cuisine_Tobii_{video_id}_cpts.npy` — Change points per participant

#### Deployment Results

| Folder | Config | Description |
|--------|--------|-------------|
| `change-point-detection-deployment/` | `ChangePointDetectionDeploymentConfig` | Deployed with estimated parameters |
| `change-point-detection-deployment-calibrated/` | `ChangePointDetectionCalibratedDeploymentConfig` | **Production** — calibrated variant |
| `gaussian-kernel-change-point-detection-deployment/` | `KernelChangePointDetectionDeploymentConfig` | Gaussian RBF kernel variant |
| `gold-change-point-detection-prototypes-deployment/` | `ChangePointDetectionPrototypesCalibratedDeploymentConfig` | Prototype-space CPD |
| `gold-change-point-detection-prototypes-deployment-calibrated/` | same, calibrated | Calibrated prototype-space CPD |

### 3.3 Clustering Deployments

**Status:** MIXED (production + exploratory)

| Folder | Status | Description |
|--------|--------|-------------|
| `clustering-deployment-faiss-cosine/` | PRODUCTION | FAISS cosine k-means (C=250) |
| `clustering-deployment-faiss-euclidean/` | PRODUCTION | FAISS euclidean k-means |
| `clustering-deployment-prototypes-v1/` | PRODUCTION | Prototype-based clustering v1 |
| `clustering-deployment-prototypes-v1-l2/` | PRODUCTION | v1 with L2 normalization |
| `clustering-deployment-prototypes-v0/` | LEGACY | Earlier prototype method |
| `clustering-deployment-kmeans-all/` | EXPLORATORY | All-data k-means baseline |
| `clustering-deployment-kmeans-task-r1/` | EXPLORATORY | Task-specific round 1 |
| `clustering-deployment-kmeans-task-r1-coarse/` | EXPLORATORY | Coarse-grained variant |
| `clustering-deployment-kmeans-exogeneous-r1/` | EXPLORATORY | Exogenous-feature variant |
| `clustering-deployment-kmeans-timestamps-all/` | EXPLORATORY | Timestamp-augmented variant |

### 3.4 Legacy: `experiments/experiments/`

**Status:** LEGACY  
Nested duplicate containing 13 experiment directories that mirror the parent structure:
- `change-point-detection-deployment/`, `change-point-detection-deployment-calibrated/`, `change-point-detection-experiment/`
- `clustering-deployment-faiss-cosine/`, `clustering-deployment-faiss-euclidean/`
- `clustering-deployment-kmeans-all/`, `clustering-deployment-kmeans-*` (5 variants)
- `clustering-deployment-prototypes-v0/`, `clustering-deployment-prototypes-v1/`, `clustering-deployment-prototypes-v1-l2/`

Likely created when the output path root was accidentally set to `experiments/experiments/` instead of `experiments/`.

---

## 4. Pipeline Outputs — `outputs/`

**Status:** MIXED  
**Created by:** Symbolization pipeline (`features/symbolization/main.py`)

Output path pattern:
```
outputs/{config.experiment_name}/{config.experiment_id}/{annotator_id}/round_{N}/
```

### 4.1 Symbolization Gold Outputs — `symbolization-gold/`

**Status:** PRODUCTION  

This is the largest directory. It stores pipeline state (DataFrames, visualizations) as opposed to `experiments/symbolization-gold/` which stores per-participant results.

#### Subdirectories (by experiment_id)

Same experiment_id structure as `experiments/symbolization-gold/` (see Section 3.1). Each round folder contains:

- **Visualizations:** PNG files of distance matrices, cluster compositions, dendrograms
- **Figure folders:** `figures-clusters/`, `figures-test/` for aggregated visualizations
- **Barycenter outputs:** `barycenter/` subfolder in later rounds
- **Subject-level analysis:** `clusters-subjects-rows/` in later rounds

#### State Files (at `symbolization-gold/` root)

Aggregated pipeline state from various symbolization configs:

| File Pattern | Description |
|-------------|-------------|
| `{ConfigName}_states_dataframe.pkl` | Full symbolization state (all participants) |
| `{ConfigName}_clusterdf.pkl` | Cluster-level statistics |
| `{ConfigName}_symbolization_registration_dataframe.pkl` | Registration metadata |
| `{ConfigName}_threshold_mapping_{N}_sigma.pkl` | Distance thresholds for prototype assignment |

Config names include: `SymbolicSourceCosineInferenceGoldConfig`, `SymbolicSourceEuclideanInferenceGoldConfig`, `SymbolicSourceFaissCInferenceGoldConfig`, `SymbolicSourceFaissEInferenceGoldConfig`, `SymbolicSourceGaussianRBFInferenceGoldConfig`, plus `*FullConfig` variants.

#### Aggregated Experiment DataFrame

`experiments_dataframe_symbolization_gold.pkl` (~178 MB) — consolidated results across all experiments (found in deeper round folders).

### 4.2 Change-Point Detection Outputs

| Folder | Status | Description |
|--------|--------|-------------|
| `gold-change-point-detection-prototypes-deployment/` | PRODUCTION | Contains `lambda_optimal.csv` (570 KB) |
| `gold-change-point-detection-prototypes-deployment-calibrated/` | PRODUCTION | Calibrated lambda values |
| `gold-change-point-detection-prototypes-experiment/` | PRODUCTION | Contains `results.pkl` (12.9 MB) |
| `gaussian-kernel-change-point-detection-deployment/` | PRODUCTION | Gamma estimates per annotator/round |

### 4.3 Legacy: `symbolization-v1/`

**Status:** LEGACY  
Early symbolization attempt containing `euclidean_subspace_projection/`. Superseded by `symbolization-gold/`.

### 4.4 Legacy Folders Inside `symbolization-gold/`

| Folder | Issue |
|--------|-------|
| `zsocre_iteration_1` | Typo ("zsocre" → "zscore") |
| `iteration_1` | Unnamed/unqualified iteration |
| `cuisine/` | Misplaced task-level subfolder (contains `Tobii/K_6/`, `Tobii/K_16/`) |

---

## 5. Speech Analysis — `experiments_speech/`

**Status:** EXPLORATORY  
**Created by:** Speech diarization analysis notebooks (not part of the main thesis pipeline)

| Subfolder | Description |
|-----------|-------------|
| `chronograms/` | Temporal speech activity diagrams |
| `data/` | Intermediate speech data |
| `dendrogram/` | 108 hierarchical clustering dendrograms |
| `plot_histograms/` | Speech feature distribution histograms |

---

## 6. Thumbnails — `thumbnails/`

**Status:** SUPPORT  
**Created by:** Annotation/visualization tools for the pigeon annotation interface

| Folder | Status | Description |
|--------|--------|-------------|
| `thumbnails-24022025/` | PRODUCTION | Latest thumbnails (42 participant folders) |
| `thumbnails-27112024/` | SUPPORT | Previous version (40 folders) |
| `thumbnails-27022024/` | SUPPORT | Earliest version (40 folders) |
| `thumbnails-cheetah/` | LEGACY | Non-cuisine task (cheetah) |
| `thumbnails-pomme/` | LEGACY | Non-cuisine task (pomme/apple) |

---

## Appendix A: Production vs Legacy Classification

### Production Folders

| Path | Pipeline Stage |
|------|----------------|
| `cuisine/*/Tobii/` | Feature extraction |
| `cuisine/*/GoPro1/` | Hand landmarks |
| `dataframes/annotations/` | Prototype annotation |
| `dataframes/clinical/` | Clinical analysis |
| `dataframes/quality-control/` | Data validation |
| `experiments/symbolization-gold/faissc_inference_symbolization/` | Final symbolization |
| `experiments/symbolization-gold/faissc_refinement_symbolization/` | Prototype refinement |
| `experiments/symbolization-gold/faissc_post_hf_symbolization/` | Post-annotation inference |
| `experiments/change-point-detection-deployment-calibrated/` | Calibrated CPD |
| `experiments/clustering-deployment-faiss-cosine/` | FAISS clustering |
| `outputs/symbolization-gold/` | Pipeline state |
| `outputs/gold-change-point-detection-prototypes-deployment/` | Lambda values |

### Legacy / Exploratory Folders

| Path | Reason |
|------|--------|
| `cuisine/experiments/` | Misplaced experiment folder |
| `experiments/experiments/` | Nested duplicate |
| `experiments/clustering-deployment-prototypes-v0/` | Superseded by v1 |
| `experiments/clustering-deployment-kmeans-all/` | Baseline experiment |
| `experiments/clustering-deployment-kmeans-task-r1-coarse/` | Exploratory variant |
| `experiments/clustering-deployment-kmeans-exogeneous-r1/` | Exploratory variant |
| `experiments/clustering-deployment-kmeans-timestamps-all/` | Exploratory variant |
| `outputs/symbolization-v1/` | Superseded by symbolization-gold |
| `outputs/symbolization-gold/zsocre_iteration_1` | Typo variant |
| `outputs/symbolization-gold/iteration_1` | Unnamed/unqualified |
| `outputs/symbolization-gold/cuisine/` | Misplaced subfolder |
| `experiments_speech/` | Not in main pipeline |
| `thumbnails-cheetah/` | Non-cuisine task |
| `thumbnails-pomme/` | Non-cuisine task |
| `dataframes/backup_lego_cake_table_post_conversion.csv` | Lego task (non-cuisine) |
| `dataframes/lego_cake_table_with_id.*` | Lego task (non-cuisine) |
| `dataframes/fix_DUMJEA.*` | One-off correction |

---

## Appendix B: File Naming Conventions

### Participant Folders
```
G{group_id}_{C|P}{participant_id}_{trigram}_{ddmmyyyy}/
```

### Feature Files (inside modality folders)
```
{feature_type}_{method}_{video_name}.{ext}
```
- `video_representations_VideoMAEv2_{video}.npy`
- `hand_landmarks_mediapipe_{video}.json`
- `skeleton_landmarks_{video}.json`
- `speech_recognition_diarization_whisperx_{video}.json`
- `speech_representations_multilingual_{video}.npy`

### Flag Files
```
.{video_name}_{feature_type}_flag.txt
```

### Experiment Folder Hierarchy
```
{experiment_name}/{experiment_id}/{annotator_id}/round_{N}/
```

### Annotation Files
```
round_{N}_prototypes_K_{k}.csv
```

### Persistent Metadata Backups
```
{hostname}_dataset_df.csv
gold_dataset_df_backup_{YYYYMMDD}.csv
```

---

## Appendix C: Pipeline-to-Folder Mapping

### Data Preprocessing and Feature Extraction

| Pipeline Step | Input | Output Location |
|---------------|-------|-----------------|
| Video feature extraction (VideoMAE-v2) | Raw .mp4 | `cuisine/*/Tobii/video_representations_*.npy` |
| Hand landmarks (MediaPipe) | Raw .mp4 | `cuisine/*/Tobii/hand_landmarks_*.json` |
| Skeleton pose (MediaPipe) | Raw .mp4 | `cuisine/*/Tobii/skeleton_landmarks_*.json` |
| Speech recognition (WhisperX) | Raw .wav | `cuisine/*/Tobii/speech_recognition_*.json` |
| Speech embeddings (E5-Large) | WhisperX output | `cuisine/*/Tobii/speech_representations_*.npy` |
| Quality control | All features | `dataframes/quality-control/` |
| Metadata generation | All features | `dataframes/persistent_metadata/` |

### Recursive Prototyping and Temporal Segmentation

| Pipeline Step | Config Class | Output Location |
|---------------|-------------|-----------------|
| Initial clustering (full data) | `ClusteringDeploymentKmeansFullConfig` | `experiments/symbolization-gold/init_clustering_*/` |
| Kernel comparison (4 metrics) | `SymbolicSource*GoldConfig` | `experiments/symbolization-gold/{cosine,euclidean,faissc,faisse}_iteration_1/` |
| CPD parameter tuning | `ChangePointDetectionExperimentConfig` | `experiments/change-point-detection-experiment/` (141 UUID runs) |
| CPD deployment | `ChangePointDetectionCalibratedDeploymentConfig` | `experiments/change-point-detection-deployment-calibrated/` |
| Post-annotation inference | `SymbolicSourceInferenceCompleteGoldConfig` | `experiments/symbolization-gold/faissc_post_hf_symbolization/` |
| Prototype refinement (8 rounds) | `SymbolicSourceInferenceRefinementGoldConfig` | `experiments/symbolization-gold/faissc_refinement_symbolization/` |
| Manual annotation (pigeon) | — | `dataframes/annotations/pigeon-annotations/symbolization-gold/` |
| Final inference | `SymbolicSourceInferenceGoldConfig` | `experiments/symbolization-gold/faissc_inference_symbolization/` |
| Pipeline state & visualizations | all symbolic configs | `outputs/symbolization-gold/` |

### Barycenter Averaging and Clinical Analysis

| Pipeline Step | Input | Output Location |
|---------------|-------|-----------------|
| TWE/eShapeDTW distance matrices | Symbolic sequences | `experiments/symbolization-gold/faissc_inference_symbolization/samperochon/round_8/D_*.npy` |
| DBA barycenter computation | Distance matrices | `outputs/symbolization-gold/faissc_inference_symbolization/samperochon/round_8/barycenter/` |
| Clinical group comparisons | Clinical CSVs + symbolic sequences | `dataframes/clinical/` |
