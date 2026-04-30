"""
Tests for §17.1 / §17.7 — EigenmodeProjector keyword fallback removal.

The April 10 correction established that eigenmode projection must be
output-side from real signals (press.direction × sigma_profile, or
physiology via DEAP), never from keyword matching on text content.

These tests assert that:
  - explicit_weights is now mandatory (no silent fallback);
  - the keyword scaffolding is gone, not just unreachable.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eft_system import EigenmodeProjector


def test_project_with_explicit_weights_works():
    """The output-side path: explicit_weights is provided, content is irrelevant."""
    projector = EigenmodeProjector()
    x_p, n_p = projector.project("anything", explicit_weights=[0.1, 0.2, 0.3, 0.4, 0.5])
    assert isinstance(x_p, np.ndarray)
    assert isinstance(n_p, np.ndarray)
    assert x_p.shape == (5,)
    assert n_p.shape == (5,)


def test_project_without_explicit_weights_raises():
    """Even content that previously would have keyword-matched ('grief',
    'sorrow' aren't in the old table — try words that were: 'memory',
    'leave', 'unknown') now fails loudly. The lookup is gone."""
    projector = EigenmodeProjector()
    with pytest.raises(ValueError) as exc_info:
        projector.project("grief sorrow memory leave unknown")
    assert "explicit_weights" in str(exc_info.value)


def test_keyword_affinity_attribute_does_not_exist():
    """The scaffolding is gone, not just unreachable."""
    projector = EigenmodeProjector()
    assert hasattr(projector, "keyword_affinity") is False, (
        "keyword_affinity dict still present — scaffolding not fully removed"
    )
    assert hasattr(projector, "_keyword_weights") is False, (
        "_keyword_weights method still present — scaffolding not fully removed"
    )
