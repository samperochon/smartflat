"""Generalization datasets for the barycenter method (PAPER_TODO §2).

The SDS2 clinical cohort is a single dataset with a *negative* order result (§15).
To argue external validity and robustness across diverse situations, each dataset
loader here emits the same four objects the SDS2 harness consumes — ``X_symbolic``,
``labels``, ``G``, ``D_G`` — and :func:`run_generalization_suite` then runs the
already-verified §15 order-null and §17-20 quality probes on any of them, unchanged.

Submodules (added per phase):
  - :mod:`.suite` — the reusable driver (Phase 0).
  - :mod:`.synthetic` — controlled sets: D1 exact-recovery + order-discriminative
    validity proof (Phase 1).
  - :mod:`.action_segmentation` — ground-truth action-label datasets used *directly*
    as symbols (Breakfast / 50Salads / GTEA / …; Phase 2).
  - :mod:`.bpi`, :mod:`.ucr` — cross-domain breadth (process mining, UCR/UEA; Phase 3).
"""
from .suite import run_generalization_suite

__all__ = ['run_generalization_suite']
