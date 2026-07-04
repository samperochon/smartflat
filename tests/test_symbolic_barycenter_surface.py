"""Guard the curated package-level re-export surface (arc-audit Phase 3, Kickoff O).

``symbolic_barycenter/__init__.py`` now re-exports the F-arc analysis API at package
level (submodules + the baseline/method surface + the harness entry points). These tests
assert every name advertised in ``__all__`` actually resolves, that the package imports
without a cycle, and that the baseline surface stays in lock-step with ``baselines.__all__``.
"""

import importlib

import smartflat.features.symbolic_barycenter as p


def test_all_names_resolve():
    """Every name in ``__all__`` is a real attribute (no dangling advertisement)."""
    missing = [n for n in p.__all__ if not hasattr(p, n)]
    assert not missing, f"__all__ advertises unresolved names: {missing}"


def test_representative_api_reachable():
    """Spot-check headline symbols are importable straight from the package."""
    from smartflat.features.symbolic_barycenter import (  # noqa: F401
        barycenter_fgw,
        default_baseline_methods,
        evaluate_baselines,
        embed_symbolic_to_real,
        RTWE_NU,
        RTWE_LMBDA,
        score_barycenter_quality,
        order_information,
        evaluate_incremental_structure,
        build_g28_ground_cost,
    )
    assert callable(barycenter_fgw)
    assert callable(score_barycenter_quality)


def test_baseline_surface_in_lockstep():
    """The re-exported baseline surface equals baselines' public (non-underscore) names."""
    from smartflat.features.symbolic_barycenter import baselines
    public = [n for n in baselines.__all__ if not n.startswith("_")]
    for n in public:
        assert hasattr(p, n), f"baselines public name {n!r} not re-exported at package level"
        assert getattr(p, n) is getattr(baselines, n)


def test_no_import_cycle():
    """A fresh import of the package (and its shim) succeeds without a cycle."""
    m = importlib.import_module("smartflat.features.symbolic_barycenter")
    assert m.baselines is importlib.import_module(
        "smartflat.features.symbolic_barycenter.baselines")
