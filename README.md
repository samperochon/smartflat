---
noteId: "01c08e702fc711f1b60c85121fd90d06"
tags: []

---

# Smartflat

Smartflat is a Python research framework for multimodal video analysis in healthcare settings, developed as part of the **SDS2 study** (Smartflat for Dysexecutive Syndrome Stratification) at Hopital National d'Instruction des Armees Percy, France. It processes video recordings from multiple camera sources (GoPro, Tobii eye-tracking glasses) to study activities of daily living, with a focus on detecting and characterizing dysexecutive syndromes through computational behavioral analysis.

## Installation

Requires **Python >= 3.10**.

```bash
pip install -e ".[all,dev]"
```

To install only specific modalities, use optional extras:

| Extra | Dependencies | Use case |
|-------|-------------|----------|
| `video` | decord | Video frame decoding |
| `audio` | librosa | Audio/speech processing |
| `pose` | mediapipe | Hand landmarks and body pose |
| `clustering` | coclust, lempel_ziv_complexity, pyentrp | Co-clustering and complexity metrics |
| `stats` | pingouin | Statistical testing |
| `annotation` | pigeon-jupyter | Interactive prototype annotation |
| `web` | flask, jupyter_dash | Web-based visualization |
| `all` | All of the above | Full installation |
| `dev` | pytest, nbstripout, pre-commit | Development tools |

```bash
# Example: install core + video + audio only
pip install -e ".[video,audio]"
```

## Data Setup

Set the data root directory before running any pipeline:

```bash
export SMARTFLAT_DATA_ROOT=/path/to/data-gold-final
```

The pipeline locates data via `get_data_root()` (in `smartflat/utils/utils_paths.py`), which checks `SMARTFLAT_DATA_ROOT` first, then falls back to host-specific paths. See [DATA_INVENTORY.md](DATA_INVENTORY.md) for the full directory structure and contents.

## Notebooks

All notebooks are in `notebooks/` and follow the pipeline order:

| # | Filename | Description | Chapter |
|---|----------|-------------|---------|
| 00 | `00_data_overview.ipynb` | Cohort statistics and dataset summary | — |
| 01 | `01_data_preprocessing.ipynb` | Raw recordings to consolidated metadata | Ch. 4 |
| 02 | `02_feature_extraction.ipynb` | VideoMAE-v2, WhisperX, MediaPipe extraction | Ch. 4 |
| 03 | `03_recursive_prototyping.ipynb` | Cosine k-means, manual annotation, HAC consolidation | Ch. 5 |
| 04 | `04_temporal_segmentation.ipynb` | Kernel change-point detection (PELT) with slope heuristic | Ch. 5 |
| 05 | `05_symbolic_representation.ipynb` | Prototypes + segments assembled into symbolic sequences | Ch. 5 |
| 06 | `06_barycenter_averaging.ipynb` | Temporal-Wasserstein distance + DBA barycenter averaging | Ch. 6 |
| 07 | `07_clinical_analysis.ipynb` | Group comparisons: Control vs TBI vs RIL | Ch. 6 |
| 08 | `08_thesis_figures.ipynb` | Thesis figure reproduction (stub) | Ch. 4–6 |

See [NOTEBOOK_ARCHIVE.md](NOTEBOOK_ARCHIVE.md) for provenance mapping to the thesis-era source notebooks.

## Scientific Context

### Study Design

- **Cohort:** 162 administrations from 122 unique participants
  - 26 healthy controls, 59 traumatic brain injury (TBI), 37 right-hemisphere stroke (RIL)
- **Task:** Cooking task (baking chocolate cake per recipe)
- **Recording:** 3 GoPro cameras (60 fps) + Tobii Pro Glasses 2 (eye-tracking at 100 Hz, video at 25 fps)
- **Additional modalities:** Hand landmarks (MediaPipe), body pose (MediaPipe), speech (WhisperX)

### Five Main Contributions

| # | Contribution | Chapter | Description |
|---|-------------|---------|-------------|
| 1 | Data preprocessing | Ch. 4 | Raw multimodal recordings to VideoMAE-v2 latent sequences (D=1408) |
| 2 | Recursive prototyping | Ch. 5 | Iterative cosine k-means (P=8 rounds, C=100), visual probing, HAC consolidation yielding ~55 validated prototypes |
| 3 | Temporal segmentation | Ch. 5 | Kernel change-point detection (PELT) with slope heuristic yielding ~274 segments per sequence |
| 4 | Barycenter averaging | Ch. 6 | Temporal-Wasserstein (TWE) distance + adapted DBA for symbolic sequences |
| 5 | Clinical analysis | Ch. 6 | Group comparisons (Control/TBI/RIL), temporal distributions, descriptive statistics |

## Running Tests

```bash
python -m pytest tests/
```

## Citation and Thesis Reference

This framework supports the scientific contributions detailed in Sam Perochon's PhD thesis (Chapters 4–6). LaTeX sources for the thesis chapters and the published paper on recursive prototyping are available in `associated-papers/`.

The codebase was refactored from the thesis-era `smartflat-thesis` repository (archived on GitHub, tag `v-thesis-final`).

## See Also

- [CLAUDE.md](CLAUDE.md) — Developer reference: architecture, package structure, configuration patterns, development conventions
- [DATA_INVENTORY.md](DATA_INVENTORY.md) — Complete inventory of the `data-gold-final/` data directory
- [NOTEBOOK_ARCHIVE.md](NOTEBOOK_ARCHIVE.md) — Provenance mapping from thesis-era notebooks to current notebooks
