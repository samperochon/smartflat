"""Pipeline engine: clustering, change point detection, and model building.

Core computational modules for the SmartFlat pipeline (Ch. 5):
- ``builders``: Factory functions for models, metrics, and datasets.
- ``clustering``: Cosine k-means via faiss for recursive prototyping.
- ``change_point_detection``: KCP/PELT temporal segmentation.
- ``fit_and_solve_cpts_curves``: Bi-scale penalty selection via slope heuristic.
- ``clustering_evaluation``: Score aggregation for clustering experiments.
- ``pca_model_computation``: PCA model fitting for dimensionality reduction.
"""
