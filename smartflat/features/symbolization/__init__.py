"""Symbolic representation pipeline for video activity sequences.

Implements the full pipeline from Chapter 5 of the thesis:
- Recursive prototyping (Section 5.2): ``main.py``, ``utils.py``
- HAC consolidation (Section 5.3): ``co_clustering.py``
- Temporal segmentation (Section 5.4): via ``engine.change_point_detection``
- Symbolic representation (Section 5.5): ``inference.py``, ``main.py``
- Temporal distributions (Section 5.3): ``temporal_distributions_estimation.py``
- Visualization: ``visualization.py``, ``visualization_prototypes.py``
"""
