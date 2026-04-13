---
noteId: "4ba1c1f0303811f1b60c85121fd90d06"
tags: []

---

# Contributing

Guide for lab members contributing code or analysis to the Smartflat project.

## Development setup

```bash
# Editable install with all dependencies
pip install -e ".[all,dev]"

# Install pre-commit hooks (strips notebook outputs on commit)
pre-commit install

# Run the test suite
python -m pytest tests/
```

## Code style

### Imports

Always use absolute package imports:

```python
# Good
from smartflat.utils.utils_paths import get_data_root
from smartflat.engine.clustering import cosine_kmeans

# Bad — never manipulate sys.path
import sys; sys.path.append("..")
```

### Paths

Never hardcode data paths. Use the environment variable and path utilities:

```python
from smartflat.utils.utils_paths import get_data_root

data_root = get_data_root()  # reads SMARTFLAT_DATA_ROOT
```

### Configuration

All experiment parameters belong in config classes (`smartflat/configs/`), not hardcoded in scripts or notebooks:

```python
from smartflat.configs.clustering_config import ClusteringConfig

config = ClusteringConfig(n_clusters=100, d_min=0.3)
```

### Logging

Use the project logger, not `print()`, for status messages in library code:

```python
from smartflat.utils.utils_coding import get_logger
logger = get_logger(__name__)

logger.info("Processing %d videos", len(videos))
```

## Working with notebooks

### Structure

All notebooks follow a consistent pattern:

1. **Imports** — smartflat modules + standard scientific stack
2. **Parameters** — config names, annotator ID, round number, etc.
3. **Data loading** — via `get_dataset()`, `get_experiments_dataframe()`, or direct metadata
4. **Pipeline execution** — long-running steps are often commented out with notes on where/how they were run (e.g., HPC)
5. **Results & visualization** — plots, tables, statistical tests

### Conventions

- Notebooks are numbered `00`-`08` and follow the pipeline order
- **Strip outputs before committing** — the `nbstripout` pre-commit hook handles this automatically
- Keep notebooks as documentation of the pipeline. Heavy computation should live in `smartflat/` modules, not inline in cells.
- When adding a new notebook, follow the existing numbering and structure

## Adding new code

### Feature extractors

New feature extraction pipelines go in `smartflat/features/<modality>/`:

```
smartflat/features/
  video/          # VideoMAE-v2 extraction
  audio/          # WhisperX ASR + embeddings
  gaze/           # Tobii eye-tracking (incomplete)
  hands/          # MediaPipe hand landmarks
  skeleton/       # MediaPipe body pose
  symbolization/  # Recursive prototyping pipeline
  symbolic_barycenter/  # TWE + DBA barycenter
  consolidation/  # Data preprocessing scripts
```

### Configs

Add a new config class in `smartflat/configs/` inheriting from `BaseConfig`. This gives you JSON serialization for reproducibility.

### Datasets

New dataset classes go in `smartflat/datasets/`, inheriting from `SmartflatDataset` in `base_dataset.py`. The base class provides metadata management and scenario filtering.

### Utilities

General-purpose helpers go in the appropriate `smartflat/utils/utils_*.py` module. Avoid creating new utility files unless the functionality is clearly distinct from existing modules.

## Data patterns

### Directory structure

Data follows the pattern: `{data_root}/{task}/{participant_id}/{modality}/`

```
data-gold-final/
  cuisine/
    G100_P86_BAUVin_25112022/    # G{id}_P{participant}_{trigram}_{date}
      GoPro1/
      GoPro2/
      GoPro3/
      Tobii/
      Annotation/
```

See [DATA_INVENTORY.md](DATA_INVENTORY.md) for the complete directory structure.

### Metadata-driven iteration

Most pipelines iterate over a metadata DataFrame with columns: `identifier`, `task_name`, `trigram`, `participant_id`, `modality`. Use existing dataset classes or `generate_video_metadata()` from `smartflat/datasets/build.py` to construct this DataFrame.

### Flag-based processing

Feature extraction progress is tracked via hidden flag files (e.g., `.{video_name}_video_representation_flag.txt`). These contain `"success"` or `"failure"` and are checked by `parse_flag()` in `utils_paths.py` to skip already-processed files.

## Git workflow

1. **Branch from `main`** — use descriptive branch names (e.g., `feature/gaze-processing`, `fix/sync-offset`)
2. **Write clear commit messages** — focus on *why*, not *what*
3. **Keep PRs focused** — one feature or fix per PR
4. **Run tests before pushing** — `python -m pytest tests/`

## Running on HPC (Ruche)

The Ruche cluster at Universite Paris-Saclay is used for GPU-intensive jobs (feature extraction, clustering). Key setup:

```bash
# On Ruche, the data root is:
export SMARTFLAT_DATA_ROOT=/gpfs/workdir/perochons/data-gold-final

# The repo lives at:
# /gpfs/users/perochons/smartflat
```

See `notebooks/02_feature_extraction.ipynb` for SLURM submission examples and HPC-specific patterns.

## Known incomplete areas

These areas are partially implemented and available for contribution:

- **Gaze processing** — `smartflat/features/gaze/main.py` raises `NotImplementedError`. Tobii data parsing exists in `smartflat/datasets/dataset_gaze.py`.
- **Kinematics analysis** — Skeleton/hand acceleration ethograms are planned but not implemented.
- **Cross-modality synchronization** — Partially implemented in `smartflat/features/consolidation/main_synchronisation.py`.

## See also

- [QUICKSTART.md](QUICKSTART.md) — 5-minute setup guide
- [README.md](README.md) — Project overview and scientific context
- [CLAUDE.md](CLAUDE.md) — Developer reference: full architecture and package structure
- [DATA_INVENTORY.md](DATA_INVENTORY.md) — Complete data directory inventory
