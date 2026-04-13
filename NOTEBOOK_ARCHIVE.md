---
noteId: "1abce3602fc711f1b60c85121fd90d06"
tags: []

---

# Notebook Archive

This document maps the 9 current notebooks (`notebooks/NB00–NB08`) back to the ~79 archived source notebooks in the `smartflat-thesis` repository (GitHub tag `v-thesis-final`).

All paths below are relative to the archived `smartflat-thesis` repository (`smartflat-thesis-local/notebooks/`).

> **Note:** This mapping was reconstructed from filenames and notebook content. Some assignments are approximate — notebooks that explored multiple topics may appear under more than one current notebook.

---

## NB00 — Data Overview

**Current:** `00_data_overview.ipynb` — Cohort statistics and dataset summary.

| Old Notebook | Description |
|-------------|-------------|
| `init.ipynb` | Environment initialization, data path setup |
| `visualization.ipynb` | General-purpose data visualization utilities |
| `onboarding/introduction.ipynb` | Project introduction and data walkthrough |

---

## NB01 — Data Preprocessing

**Current:** `01_data_preprocessing.ipynb` — Raw recordings to consolidated metadata.

| Old Notebook | Description |
|-------------|-------------|
| `dataset-preprocessing/data-integration.ipynb` | Raw data discovery and folder integration |
| `dataset-preprocessing/dataset-testing.ipynb` | Dataset validation and integrity checks |
| `dataset-preprocessing/features-extraction.ipynb` | Feature extraction pipeline setup |
| `dataset-preprocessing/quality-control.ipynb` | Visual inspection and quality control |
| `synchronisation.ipynb` | Cross-modality audio synchronization |
| `scripts.ipynb` | General pipeline execution scripts |
| `scripts_cheetah.ipynb` | HPC execution scripts (Cheetah cluster) |
| `scripts_local.ipynb` | Local execution scripts |
| `scripts_percy.ipynb` | HPC execution scripts (Percy cluster) |
| `scripts_pomme.ipynb` | HPC execution scripts (Pomme cluster) |
| `scripts_ruche.ipynb` | HPC execution scripts (Ruche cluster) |

---

## NB02 — Feature Extraction

**Current:** `02_feature_extraction.ipynb` — VideoMAE-v2, WhisperX, MediaPipe extraction.

| Old Notebook | Description |
|-------------|-------------|
| `dataset-preprocessing/features-extraction.ipynb` | Feature extraction pipeline (shared with NB01) |
| `demo_speech_data.ipynb` | Speech/audio data exploration and WhisperX usage |
| `demo_gaze_data_analysis.ipynb` | Tobii eye-tracking data parsing and analysis |
| `demo_tmp_gaze.ipynb` | Temporary gaze data exploration |
| `dataset-processing/demo_kinematics_data_analysis.ipynb` | Hand/skeleton kinematics analysis |
| `dataset-processing/pca-computation.ipynb` | PCA dimensionality reduction on features |

---

## NB03 — Recursive Prototyping

**Current:** `03_recursive_prototyping.ipynb` — Cosine k-means, annotation, HAC consolidation.

| Old Notebook | Description |
|-------------|-------------|
| `symbolization/demo_recursive_procedure.ipynb` | Full recursive prototyping pipeline |
| `symbolization/demo_recursive_procedure-light.ipynb` | Lightweight version of recursive procedure |
| `symbolization/demo_prototypes.ipynb` | Prototype extraction and visualization |
| `symbolization/prototypes_annotator.ipynb` | Interactive prototype annotation (pigeon) |
| `symbolization/demo_prototypical_token_mining.ipynb` | Prototypical token mining exploration |
| `demo_prototypes_annotation.ipynb` | Prototype annotation workflow |
| `demo_prototypical_token_mining.ipynb` | Token mining (root-level copy) |
| `10_clustering.ipynb` | K-means clustering experiments |
| `demo_clustering.ipynb` | Clustering methods comparison |
| `dataset-processing/demo_clustering_comparison.ipynb` | Extended clustering comparison |

---

## NB04 — Temporal Segmentation

**Current:** `04_temporal_segmentation.ipynb` — Kernel change-point detection with slope heuristic.

| Old Notebook | Description |
|-------------|-------------|
| `10_change-point-detection.ipynb` | Main KCP/PELT experiments |
| `loss-optimization/analysis_change_point_space.ipynb` | Change-point parameter space analysis |
| `loss-optimization/analysis_min_size.ipynb` | Minimum segment size analysis |
| `loss-optimization/demo_change_ponts_characterization.ipynb` | Change-point characterization |
| `loss-optimization/regularisation-parameter-analysis.ipynb` | Regularization parameter (slope heuristic) |
| `symbolization/demo_temporal_segmentation.ipynb` | Temporal segmentation within symbolization pipeline |
| `symbolization/demo_edges_detection_morphological_filters.ipynb` | Edge detection / morphological filtering |

---

## NB05 — Symbolic Representation

**Current:** `05_symbolic_representation.ipynb` — Prototypes + segments assembled into symbolic sequences.

| Old Notebook | Description |
|-------------|-------------|
| `symbolization/demo_symbolization_gold.ipynb` | Gold-standard symbolization pipeline |
| `symbolization/demo_reduction_prototypes_distances.ipynb` | Prototype distance matrix reduction |
| `symbolization/demo_inference_wasserstein_subspace_projection.ipynb` | Wasserstein subspace projection for inference |
| `symbolization/demo_hungarian_matching.ipynb` | Hungarian matching between prototype sets |
| `symbolization/analysis_additional_experiments.ipynb` | Additional symbolization experiments |
| `dataset-processing/demo_temporal_distributions.ipynb` | Temporal KDE estimation |
| `dataset-processing/demo_prototypes_annotations_mapping.ipynb` | Prototype-to-annotation mapping |
| `demo_distance_matrix_reduction.ipynb` | Distance matrix reduction (root-level) |

---

## NB06 — Barycenter Averaging

**Current:** `06_barycenter_averaging.ipynb` — TWE distance + DBA barycenter averaging.

| Old Notebook | Description |
|-------------|-------------|
| `demo_rtwe_barycenter_averaging.ipynb` | Registered TWE + DBA pipeline |
| `last-mile-barycenter.ipynb` | Barycenter computation refinement |
| `last-mile-barycenter-twe.ipynb` | TWE-specific barycenter experiments |
| `last-mile-transition-matrix.ipynb` | Transition matrix and community detection |
| `demo_MVHC.ipynb` | Multi-View Hierarchical Clustering |
| `dataset-processing/demo_space_hyperbolicity_analysis.ipynb` | Hyperbolicity analysis of distance spaces |

---

## NB07 — Clinical Analysis

**Current:** `07_clinical_analysis.ipynb` — Group comparisons: Control vs TBI vs RIL.

| Old Notebook | Description |
|-------------|-------------|
| `demo_clinical_data.ipynb` | Clinical data loading and exploration |
| `healhcare-analysis/demo_clinical_usage.ipynb` | Clinical data usage patterns |
| `healhcare-analysis/demo_group_comparison_paper.ipynb` | Group comparison for paper |
| `healhcare-analysis/demo_segments_durations_analysis.ipynb` | Segment duration analysis by group |
| `healhcare-analysis/analysis_gaze_paper.ipynb` | Gaze analysis for paper |
| `statistics.ipynb` | Statistical testing (MWU, LME) |
| `dataset-processing/demo_qualitative_assessments.ipynb` | Qualitative assessment coding |
| `reports-experiments/demo_group_comparison-ressources.ipynb` | Group comparison resources |
| `reports-experiments/demo_qualitative_assessments-ressources.ipynb` | Qualitative assessment resources |
| `reports-experiments/demo_qualitative_assessments.ipynb` | Qualitative assessments (report version) |

---

## NB08 — Figures

**Current:** `08_figures.ipynb` — Figure reproduction (stub).

| Old Notebook | Description |
|-------------|-------------|
| `thesis-content/thesis-figures.ipynb` | Main thesis figure generation |
| `thesis-content/thesis-recipes-utils.ipynb` | Utility functions for thesis figures |
| `symbolization/demo_prototypes_visual_figures.ipynb` | Prototype visual figures for thesis |
| `symbolization/worker_prototypes_visual_figures.ipynb` | Batch generation of prototype figures |

---

## Unmapped Notebooks

The following archived notebooks were not consolidated into any current notebook. They fall into three categories:

### Infrastructure / HPC Setup

These notebooks contained machine-specific setup and execution scripts. Their content was either absorbed into NB01's scripts section or is no longer needed:

- `warming-up-cheetah.ipynb` — Cheetah cluster warm-up
- `warming-up-local.ipynb` — Local environment warm-up

### Temporary / Buffer Notebooks

Working notebooks used for intermediate report generation, not carrying persistent analysis:

- `reports-experiments/buffer_plots.ipynb`
- `reports-experiments/buffer-plots-2.ipynb`
- `reports-experiments/report-bufffer.ipynb`
- `reports-experiments/demo_spacetime_fusion_paper.ipynb`
- `reports-experiments/demo_prototypes_annotation-ressources-v1.ipynb`

### Other

- `thibaut-LT-normalization-experiment-results.ipynb` — Collaborator experiment (Thibaut)
- `dataset-processing/demo_vision_annotations.ipynb` — Vision-based annotation exploration
- `dataset-processing/follow-up-fitting.ipynb` — Follow-up model fitting
- `dataset-processing/run_deployment.ipynb` — Deployment execution
- `dataset-processing/demo_K_analsys-ressources.ipynb` — K-analysis resources
- `onboarding/demo_annotations.ipynb` — Onboarding annotation demo
- `supervised_work/demo_annotations.ipynb` — Supervised annotation work
- `demo_annotations.ipynb` — Annotation demo (root-level)
