"""Shared fixtures for distance module tests."""

import numpy as np
import pytest


@pytest.fixture
def D5():
    """5x5 symmetric distance matrix with zero diagonal (prototypes 0-4)."""
    rng = np.random.RandomState(42)
    D = rng.rand(5, 5)
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0.0)
    return D


@pytest.fixture
def D28():
    """28x28 symmetric ground cost (G=28 vocabulary), zero diagonal, C-contiguous."""
    rng = np.random.RandomState(7)
    D = rng.rand(28, 28)
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0.0)
    return np.ascontiguousarray(D)
