# Smartflat

Smartflat is a Python research framework for multimodal video analysis in healthcare settings, developed as part of the **SDS2 study** (Smartflat for Dysexecutive Syndrome Stratification) at Hopital National d'Instruction des Armees Percy, France. It processes video recordings from multiple camera sources (GoPro, Tobii eye-tracking glasses) to study activities of daily living, with a focus on detecting and characterizing dysexecutive syndromes through computational behavioral analysis.

## Installation

Requires **Python >= 3.10**.

```bash
pip install -e ".[all,dev]"
```


## Data Setup

Set the data root directory before running any pipeline:

```bash
export SMARTFLAT_DATA_ROOT=/path/to/data-gold-final
```

The pipeline locates data via `get_data_root()` (in `smartflat/utils/utils_paths.py`), which checks `SMARTFLAT_DATA_ROOT` first, then falls back to host-specific paths. See [DATA_INVENTORY.md](DATA_INVENTORY.md) for the full directory structure and contents.

## Notebooks

All notebooks are in `notebooks/` and follow the pipeline order:

| # | Filename | Description |
|---|----------|-------------|
| 00 | `00_data_overview.ipynb` | Cohort statistics and dataset summary |
| 01 | `01_data_preprocessing.ipynb` | Raw recordings to consolidated metadata |
| 02 | `02_feature_extraction.ipynb` | VideoMAE-v2, WhisperX, MediaPipe extraction |
| 03 | `03_recursive_prototyping.ipynb` | Cosine k-means, manual annotation, HAC consolidation |
| 04 | `04_temporal_segmentation.ipynb` | Kernel change-point detection (PELT) with slope heuristic |
| 05 | `05_symbolic_representation.ipynb` | Prototypes + segments assembled into symbolic sequences |
| 06 | `06_barycenter_averaging.ipynb` | Temporal-Wasserstein distance + DBA barycenter averaging |
| 07 | `07_clinical_analysis.ipynb` | Group comparisons: Control vs TBI vs RIL |
| 08 | `08_figures.ipynb` | Figure reproduction (stub) |

See [NOTEBOOK_ARCHIVE.md](NOTEBOOK_ARCHIVE.md) for provenance mapping from the archived source notebooks.

## Scientific Context

### Study Design

- **Cohort:** 162 administrations from 122 unique participants
  - 26 healthy controls, 59 traumatic brain injury (TBI), 37 right-hemisphere stroke (RIL)
- **Task:** Cooking task (baking chocolate cake per recipe)
- **Recording:** 3 GoPro cameras (60 fps) + Tobii Pro Glasses 2 (eye-tracking at 100 Hz, video at 25 fps)
- **Additional modalities:** Hand landmarks (MediaPipe), body pose (MediaPipe), speech (WhisperX)

### Five Main Contributions

| # | Contribution | Description |
|---|-------------|-------------|
| 1 | Data preprocessing | Raw multimodal recordings to VideoMAE-v2 latent sequences (D=1408) |
| 2 | Recursive prototyping | Iterative cosine k-means (P=8 rounds, C=100), visual probing, HAC consolidation yielding ~55 validated prototypes |
| 3 | Temporal segmentation | Kernel change-point detection (PELT) with slope heuristic yielding ~274 segments per sequence |
| 4 | Barycenter averaging | Temporal-Wasserstein (TWE) distance + adapted DBA for symbolic sequences |
| 5 | Clinical analysis | Group comparisons (Control/TBI/RIL), temporal distributions, descriptive statistics |

## Running Tests

```bash
python -m pytest tests/
```

## Citation

LaTeX sources for the associated publications and the published paper on recursive prototyping are available in `associated-papers/`.

## See Also

- [QUICKSTART.md](QUICKSTART.md) — 5-minute setup guide for new lab members
- [CONTRIBUTING.md](CONTRIBUTING.md) — Code style, development workflow, how to add new features
- [CLAUDE.md](CLAUDE.md) — Developer reference: architecture, package structure, configuration patterns, development conventions
- [DATA_INVENTORY.md](DATA_INVENTORY.md) — Complete inventory of the `data-gold-final/` data directory
- [NOTEBOOK_ARCHIVE.md](NOTEBOOK_ARCHIVE.md) — Provenance mapping from archived source notebooks
