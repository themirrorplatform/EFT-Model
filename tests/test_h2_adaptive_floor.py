"""
Tests for §17.4 — H2 adaptive update with survival-floor clamps.

Architectural rule: alpha and beta must be able to move both up and down
based on valence pressure (mirroring H1's mu1 dynamics for the magnitude
channel), with hard clamps at the documented survival floors:
  alpha >= 0.05  (approach orientation cannot be extinguished)
  beta  >= 0.01  (avoidance orientation cannot be extinguished)

Pre-fix, update_H2 was monotone non-decreasing — the floor held vacuously
because there was no downward dynamic. These tests assert that H2 now
responds in both directions and clamps at the floors.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eft_encounter_env import ConditionFieldState, HistoryUpdater


def test_h2_responds_up_to_positive_valence():
    """Under sustained full-positive valence: alpha rises, beta falls.
    Pre-fix, alpha would rise but beta would NOT fall. This is the new
    behavior."""
    H = ConditionFieldState()
    updater = HistoryUpdater()
    assert H.alpha == 0.5 and H.beta == 0.5
    for _ in range(10):
        updater.update_H2(H, eps2=1.0, r2_current=0.0, eta=0.3)
    assert H.alpha > 0.5, f"alpha should rise toward target=1, got {H.alpha}"
    assert H.beta < 0.5, f"beta should fall toward target=0, got {H.beta}"


def test_h2_responds_down_when_valence_reverses():
    """Core fix: under sustained negative valence, previously-elevated
    alpha must come back down. Pre-fix code would leave alpha frozen at
    the elevated value because there was no downward dynamic."""
    H = ConditionFieldState()
    updater = HistoryUpdater()
    for _ in range(20):
        updater.update_H2(H, eps2=1.0, r2_current=0.0, eta=0.3)
    alpha_high = H.alpha
    assert alpha_high > 0.7, f"setup expected alpha > 0.7, got {alpha_high}"
    for _ in range(20):
        updater.update_H2(H, eps2=-1.0, r2_current=0.0, eta=0.3)
    assert H.alpha < alpha_high, (
        f"alpha must come DOWN under sustained negative valence: "
        f"alpha_high={alpha_high}, alpha_after={H.alpha}"
    )


def test_h2_clamps_at_alpha_floor():
    """Sustained full-negative valence drives alpha toward zero, but
    the survival floor at 0.05 must hold."""
    H = ConditionFieldState()
    updater = HistoryUpdater()
    for _ in range(200):
        updater.update_H2(H, eps2=-1.0, r2_current=0.0, eta=0.5)
    assert H.alpha >= 0.05, (
        f"alpha must clamp at survival floor 0.05, got {H.alpha}"
    )


def test_h2_clamps_at_beta_floor():
    """Sustained full-positive valence drives beta toward zero, but
    the survival floor at 0.01 must hold."""
    H = ConditionFieldState()
    updater = HistoryUpdater()
    for _ in range(200):
        updater.update_H2(H, eps2=1.0, r2_current=0.0, eta=0.5)
    assert H.beta >= 0.01, (
        f"beta must clamp at survival floor 0.01, got {H.beta}"
    )
