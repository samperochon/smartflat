---
noteId: "3203aec0303811f1b60c85121fd90d06"
tags: []

---

# Quickstart

Get up and running with Smartflat in 5 minutes.

## Prerequisites

- **Python >= 3.10**
- **Access to `data-gold-final`** — the processed dataset (~X GB). Ask your supervisor for the location on the lab's external drive or network storage.

## 1. Clone and install

```bash
git clone <repo-url>
cd smartflat
pip install -e ".[all,dev]"
pre-commit install
```

The `[all,dev]` extra installs all modality dependencies plus development tools. For a lighter install, pick only what you need (see [README.md](README.md#installation) for the full list of extras).

## 2. Set the data root

Smartflat needs to know where `data-gold-final` lives. Set the environment variable:

```bash
export SMARTFLAT_DATA_ROOT=/path/to/data-gold-final
```

Common locations:

| Machine | Path |
|---------|------|
| Lab MacBook (external drive) | `/Volumes/Smartflat/data-gold-final` |
| HPC (Ruche) | `/gpfs/workdir/perochons/data-gold-final` |
| Lab desktop (PCNomad) | `/media/sam/Smartflat/data-gold-final` |

Add the export to your shell profile (`~/.zshrc` or `~/.bashrc`) so it persists across sessions.

## 3. Verify your setup

```python
from smartflat.utils.utils_paths import get_data_root
print(get_data_root())  # Should print your data-gold-final path
```

Then run the test suite:

```bash
python -m pytest tests/
```

## 4. Run your first notebook

Open `notebooks/00_data_overview.ipynb` in Jupyter and run all cells. This notebook loads the cohort metadata and produces summary statistics — no heavy computation, no GPU needed. If everything runs without errors, your setup is complete.

```bash
jupyter notebook notebooks/00_data_overview.ipynb
```

## Notebook roadmap

The notebooks follow the analysis pipeline in order:

| # | Notebook | What it does |
|---|----------|-------------|
| 00 | `00_data_overview.ipynb` | Cohort statistics and dataset summary |
| 01 | `01_data_preprocessing.ipynb` | Raw recordings to consolidated metadata |
| 02 | `02_feature_extraction.ipynb` | VideoMAE-v2, WhisperX, MediaPipe extraction |
| 03 | `03_recursive_prototyping.ipynb` | Cosine k-means clustering and prototype annotation |
| 04 | `04_temporal_segmentation.ipynb` | Kernel change-point detection (PELT) |
| 05 | `05_symbolic_representation.ipynb` | Prototypes + segments assembled into symbolic sequences |
| 06 | `06_barycenter_averaging.ipynb` | Temporal-Wasserstein distance and barycenter averaging |
| 07 | `07_clinical_analysis.ipynb` | Group comparisons: Control vs TBI vs RIL |
| 08 | `08_figures.ipynb` | Figure reproduction (stub) |

Start with NB00, then work through them in order to follow the full pipeline.

## Where to go next

- [README.md](README.md) — Project overview and scientific context
- [CONTRIBUTING.md](CONTRIBUTING.md) — Code style, development workflow, how to add new features
- [CLAUDE.md](CLAUDE.md) — Developer reference: architecture, package structure, configuration patterns
- [DATA_INVENTORY.md](DATA_INVENTORY.md) — Full directory structure of `data-gold-final`
- [NOTEBOOK_ARCHIVE.md](NOTEBOOK_ARCHIVE.md) — Provenance mapping from archived source notebooks
