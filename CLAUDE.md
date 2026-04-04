---
noteId: "d6b7dba02f8711f1b60c85121fd90d06"
tags: []

---

# CLAUDE.md — Smartflat Project

## Project Overview

Smartflat is a Python research and analysis framework for multimodal video analysis in healthcare settings, developed as part of the SDS2 study (Smartflat for Dysexecutive Syndrome Stratification) at Hôpital National d'Instruction des Armées Percy, France.

The framework processes video recordings from multiple camera sources (GoPro, Tobii eye-tracking glasses) to study activities of daily living, with a focus on detecting and characterizing dysexecutive syndromes through computational behavioral analysis.

**Thesis reference:** The scientific foundations are detailed in Sam Perochon's PhD thesis, chapters 4-6 (LaTeX sources in `associated-papers/`). A published paper on recursive prototyping is in `associated-papers/paper-chapter-5/`.

## Scientific Context

### Study Design (Chapter 4)
- **Cohort:** 162 administrations from 122 unique participants (26 controls, 59 TBI, 37 RIL)
- **Task:** Cooking task (baking chocolate cake per recipe)
- **Recording:** 3 GoPro cameras (60fps) + Tobii Pro Glasses 2 (eye-tracking at 100Hz, video at 25fps)
- **Additional modalities:** Hand landmarks (MediaPipe), body pose (MediaPipe), speech (WhisperX)

### Five Main Contributions

| # | Contribution | Chapter | Key Modules |
|---|---|---|---|
| 1 | **Data preprocessing**: Raw multimodal recordings → VideoMAE-v2 latent sequences (D=1408) | Ch. 4 | `features/video/`, `features/consolidation/`, `datasets/` |
| 2 | **Recursive prototyping**: Iterative cosine k-means (P=8 rounds, C=100), visual probing, HAC consolidation → ~55 validated prototypes | Ch. 5 | `features/symbolization/`, `engine/clustering.py` |
| 3 | **Temporal segmentation**: Kernel change-point detection (PELT) with slope heuristic → ~274 segments/sequence | Ch. 5 | `engine/change_point_detection.py`, `configs/change_points_config.py` |
| 4 | **Barycenter averaging**: Temporal-Wasserstein TWE distance + adapted DBA for symbolic sequences | Ch. 6 | `features/symbolic_barycenter/`, `engine/distances/` |
| 5 | **Clinical analysis**: Group comparisons (Control/TBI/RIL), temporal distributions, descriptive statistics | Ch. 6 | `utils/utils_clinical.py`, notebooks |

### Ongoing / Incomplete Work
- **Gaze processing**: `features/gaze/main.py` has `NotImplementedError` — Tobii data parsing exists in `datasets/dataset_gaze.py`
- **Kinematics analysis**: Skeleton/hand acceleration ethograms — planned but not implemented
- **Clinical data integration**: Mostly in notebooks, minimal in API
- **Cross-modality synchronization**: In-progress, partially in `features/consolidation/main_synchronisation.py`
- **Annotation automation**: Partial, reducing manual annotation burden

## Architecture

### Package Structure (`smartflat/`)

- **`configs/`**: Configuration system based on `BaseConfig` with JSON serialization
  - `smartflat_config.py`: VideoMAE params (segment_length=16, delta_t=8)
  - `change_points_config.py`: KCP/PELT parameters, slope heuristic
  - `clustering_config.py`: K-means parameters (C=100, d_min=0.3)
  - `symbolic_config.py` / `symbolic_config_gold.py`: Symbolization pipeline configs

- **`datasets/`**: PyTorch Dataset classes
  - `base_dataset.py`: `SmartflatDataset` with metadata management and scenario filtering
  - `dataset_video_representations.py`: VideoBlock, Prototypes, Symbols, Segments datasets
  - `dataset_multimodal.py`: Combines Tobii + GoPro modalities
  - `dataset_gaze.py`, `dataset_hands.py`, `dataset_skeleton.py`, `dataset_speech.py`
  - `build.py`: `generate_video_metadata()` — discovers videos, matches to features, creates metadata
  - `filter.py`: Gold data filtering, cohort definition
  - `quality_control.py`: Visual inspection corrections, modality swap fixes

- **`features/`**: Feature extraction and analysis pipelines
  - `consolidation/`: 17 data preprocessing scripts (registration, metadata, sync, collation, distribution)
  - `video/`: VideoMAE-v2 extraction (ViT-Giant-Patch14, D=1408)
  - `audio/`: WhisperX ASR + multilingual-e5-large embeddings
  - `gaze/`: Tobii eye-tracking (INCOMPLETE)
  - `hands/` + `hands_processing/`: MediaPipe hand landmarks + temporal tracking
  - `skeleton/`: MediaPipe pose landmarks
  - `symbolization/`: Recursive prototyping pipeline (Ch. 5) — clustering, annotation, HAC, inference
  - `symbolic_barycenter/`: TWE distance + DBA barycenter averaging (Ch. 6)

- **`engine/`**: Analysis engines
  - `change_point_detection.py`: KCP/PELT with slope heuristic
  - `clustering.py`: Cosine k-means (faiss), kernel k-means
  - `clustering_evaluation.py`: Silhouette, DBI, CHI
  - `builders.py`: Model/metrics factory functions
  - `distances/`: Vendored elastic distance metrics for symbolic sequences (Ch. 6 contribution)
    - `_rtwe.py`: Registered TWE — TWE with precomputed prototype distance matrix (replaces Euclidean)
    - `_eshape_dtw.py`: Edit-Shape DTW using rTWE as inner cost
    - `_alignment_paths.py`, `_bounding_matrix.py`, `_utils.py`: Vendored from aeon (BSD-3)

- **`utils/`**: Utilities
  - `utils_io.py`: Host-aware data path detection via `get_data_root()`
  - `utils_visualization.py`: Plot helpers
  - `utils_clinical.py`: Clinical data utilities
  - `utils_coding.py`: Analysis utilities

- **`annotation_smartflat.py`**: BORIS and Vidat annotation parsing, clinical coding (A-J categories)
- **`constants.py`**: Dataset-wide constants (tasks, modalities, participant lists, feature definitions)
- **`metrics.py`**: Segmentation evaluation (F1, precision/recall, NMI, ARI)

### Data Organization

Data is organized by task and participant:
```
{SMARTFLAT_DATA_ROOT}/
  cuisine/                              # Task name
    G100_P86_BAUVin_25112022/           # G{id}_P{participant}_{trigram}_{date}
      Tobii/                            # Modality folder
      GoPro1/
      GoPro2/
      GoPro3/
      Annotation/
  dataframes/                           # Processed metadata
  experiments/                          # Experimental results
  outputs/                              # Feature extraction outputs
```

### Key Patterns

- **Host-aware paths**: `get_data_root()` returns different data paths based on machine hostname. Set `SMARTFLAT_DATA_ROOT` env var to override.
- **Metadata-driven**: Operations iterate over metadata DataFrames with columns: `identifier`, `task_name`, `trigram`, `participant_id`, `modality`
- **Flag-based processing**: Computed features tracked via flag files (e.g., `*_video_representation_flag.txt`)
- **Config-driven experiments**: All pipeline parameters in config classes for reproducibility

## Commands

```bash
# Install (development mode)
pip install -e ".[all,dev]"

# Run tests
python -m pytest tests/

# Set data root (required on new machines)
export SMARTFLAT_DATA_ROOT=/path/to/data-gold-final

# Feature extraction (on HPC)
python -m smartflat.features.video.main
python -m smartflat.features.skeleton.main
python -m smartflat.features.hands.main
```

## Development Conventions

- **Imports**: Always use `from smartflat.xxx import yyy` — no `sys.path` manipulation
- **Package install**: Use `pip install -e .` for development
- **Notebooks**: Strip outputs before committing (`nbstripout` pre-commit hook)
- **Configs**: All experiment parameters go in config classes, not hardcoded
- **Data paths**: Use `get_data_root()` or `SMARTFLAT_DATA_ROOT`, never hardcoded paths

## Refactoring Status

This repo was created from the thesis-era `smartflat-thesis` codebase (archived on GitHub, tag `v-thesis-final`).

### Completed Stages (March–April 2026)

| Stage | Scope | Status |
|-------|-------|--------|
| 0 | Foundation (package scaffolding, `.gitignore`, `pyproject.toml`) | Done |
| 1 | Archive & Migration (copy from thesis repo, clean history) | Done |
| 2 | Data Infrastructure (`utils_paths.py`, `utils_video.py`, corrections) | Done |
| 3 | Annotations (`annotation.py` deleted, `annotation_smartflat.py` documented) | Done |
| 4 | Engine + Symbolization (star imports, docstrings, 20 files cleaned) | Done |
| 5 | Barycenter + Clinical (VAME bugs fixed, 80+ unused imports removed) | Done |
| 6 | Notebooks (all 7 notebooks NB01–NB07 filled with production code) | Done |
| 7 | Polish (star imports replaced, TODOREMOVE cleaned, dead code removed) | Done |
| — | NB00 Data Overview (`00_data_overview.ipynb`, 51 cells) + `DATA_INVENTORY.md` | Done |
| 8 | Documentation (`README.md`, `NOTEBOOK_ARCHIVE.md`, CLAUDE.md update) | Done |

### Remaining

- **NB08 — Thesis Figures** (optional): Port figure-generation notebook from thesis repo (currently a stub)
