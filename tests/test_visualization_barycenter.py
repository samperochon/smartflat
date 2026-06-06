"""Smoke tests for the Chapter-6 barycenter visualization helpers.

Cover the thesis-era functions restored / added for the
``chapter_6_examples_matching``, ``chapter_6_bg`` / ``chapter_6_non_bg`` figures
and the NB06d audit plots. These are rendering smoke tests (matplotlib ``Agg``):
they assert the functions run, return the right object, accept a provided ``ax``,
do not crash on the previously-buggy ``cost_path=None`` path, and write a PNG.
"""

from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from smartflat.features.symbolic_barycenter.visualization import (
    audit_line_vs_lambda,
    audit_scatter,
    coarse_case_counts,
    compute_alignment_path,
    match_fraction,
    plot_chronogram_alignment,
    plot_cohort_barycenters,
    plot_matching_grid,
)


@pytest.fixture
def symbolic_pair():
    x = np.array([1, 1, 2, 2, 3, 3, 0, 0, 4, 4])
    y = np.array([1, 1, 2, 3, 3, 3, 0, 0, 4, 5])
    path = [(i, i) for i in range(len(x))]  # lockstep diagonal path
    return x, y, path


@pytest.fixture
def ground_cost():
    rng = np.random.RandomState(0)
    g = 6
    d = rng.rand(g, g)
    d = (d + d.T) / 2
    np.fill_diagonal(d, 0.0)
    return d


def test_alignment_returns_counter_and_no_cost_path_crash(symbolic_pair):
    x, y, path = symbolic_pair
    # do_plot=False must not raise even with cost_path=None (the old crash).
    cc = plot_chronogram_alignment(x, y, paths=path, cost_path=None, do_plot=False)
    assert isinstance(cc, Counter)
    assert sum(cc.values()) == len(path) - 1


def test_alignment_accepts_provided_ax_and_sets_match_title(symbolic_pair):
    x, y, path = symbolic_pair
    fig, ax = plt.subplots()
    cc = plot_chronogram_alignment(x, y, paths=path, ax=ax, background_values=(0,))
    assert isinstance(cc, Counter)
    assert "Match" in ax.get_title()  # "(Match: ..%)" suffix present
    plt.close(fig)


def test_match_fraction_in_unit_interval(symbolic_pair):
    x, y, path = symbolic_pair
    cc = plot_chronogram_alignment(x, y, paths=path, do_plot=False, background_values=(0,))
    coarse = coarse_case_counts(cc)
    assert set(coarse).issubset({"Match", "Background", "Addition", "Mismatch", "Other"})
    assert 0.0 <= match_fraction(cc) <= 1.0


@pytest.mark.parametrize("method", ["rtwe", "eshape"])
def test_compute_alignment_path_techniques(method, symbolic_pair, ground_cost):
    x, y, _ = symbolic_pair
    # remap symbols into the ground-cost index range
    x = x % ground_cost.shape[0]
    y = y % ground_cost.shape[0]
    path, dist = compute_alignment_path(method, x, y, D_G=ground_cost, nu=1e-3, lmbda=0.1)
    assert len(path) >= 1
    assert np.isfinite(dist)


def test_matching_grid_writes_png(tmp_path, symbolic_pair, ground_cost):
    x, y, _ = symbolic_pair
    x = x % ground_cost.shape[0]
    y = y % ground_cost.shape[0]
    panels = []
    for method in ("rtwe", "eshape"):
        path, dist = compute_alignment_path(method, x, y, D_G=ground_cost, nu=1e-3, lmbda=0.1)
        panels.append({"x": x, "y": y, "path": path,
                       "label": f"{method.upper()} distance={dist:.1f}",
                       "background_values": (0,)})
    out = tmp_path / "examples_matching.png"
    code_to_label = {i: f"cat{i}" for i in range(ground_cost.shape[0])}
    fig = plot_matching_grid(panels, n_cols=2, code_to_label=code_to_label,
                             background_values=(0,), savepath=str(out))
    assert out.exists() and out.stat().st_size > 0
    plt.close(fig)


def test_cohort_barycenters_shared_colormap_and_mask(tmp_path):
    rng = np.random.RandomState(1)
    g, t = 8, 60
    bary = {grp: rng.randint(0, g, size=t) for grp in ("Control", "TBI", "RIL")}
    out_bg = tmp_path / "bg.png"
    out_nonbg = tmp_path / "non_bg.png"
    fig_bg = plot_cohort_barycenters(bary, groups=["Control", "TBI", "RIL"],
                                     savepath=str(out_bg))
    fig_nb = plot_cohort_barycenters(bary, groups=["Control", "TBI", "RIL"],
                                     mask_background=True, background_value=0,
                                     savepath=str(out_nonbg))
    # one axis (block) per cohort, all sharing the same imshow norm/cmap.
    bg_axes = [a for a in fig_bg.axes if a.images]
    assert len(bg_axes) == 3
    norms = [a.images[0].norm for a in bg_axes]
    cmaps = [a.images[0].get_cmap() for a in bg_axes]
    assert all(n is norms[0] for n in norms)
    assert all(c is cmaps[0] for c in cmaps)
    assert out_bg.exists() and out_nonbg.exists()
    plt.close(fig_bg)
    plt.close(fig_nb)


def test_cohort_barycenters_accepts_group_stacks(tmp_path):
    rng = np.random.RandomState(3)
    g, t = 6, 50
    stacks = {"Control": rng.randint(0, g, size=(8, t)),
              "TBI": rng.randint(0, g, size=(20, t)),
              "RIL": rng.randint(0, g, size=(14, t))}
    fig = plot_cohort_barycenters(stacks, groups=["Control", "TBI", "RIL"])
    blocks = [a for a in fig.axes if a.images]
    assert len(blocks) == 3
    plt.close(fig)


def test_audit_scatter_writes_png(tmp_path):
    rng = np.random.RandomState(2)
    x = rng.rand(60)
    y = 0.6 * x + 0.2 * rng.rand(60)
    out = tmp_path / "scatter.png"
    fig = audit_scatter(x, y, "p_match", "overlap", "Audit", savepath=str(out))
    assert out.exists() and out.stat().st_size > 0
    plt.close(fig)


def test_audit_line_vs_lambda_writes_png(tmp_path):
    lam = [0.0, 1e-3, 1e-2, 0.05, 0.1, 0.2, 0.5, 1.0]
    y1 = [0.1, 0.2, 0.35, 0.5, 0.6, 0.7, 0.8, 0.85]
    y2 = [0.3, 0.32, 0.36, 0.4, 0.42, 0.45, 0.5, 0.55]
    out = tmp_path / "line.png"
    fig = audit_line_vs_lambda(lam, [y1, y2], labels=["edit frac", "aux"],
                               ylabel="fraction", title="Audit B",
                               selected_lambda=0.1, baseline=0.5, savepath=str(out))
    assert out.exists() and out.stat().st_size > 0
    plt.close(fig)
