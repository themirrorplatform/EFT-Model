"""
Tests for Phase 2.1 Level 3 — TraceLinkGraph and edge weighting.

Covers:
  - First trace has no links.
  - Temporally close encounters produce edges through the interaction layer.
  - Temporal proximity dominates (50% of weight).
  - Eigenmode misalignment reduces weight.
  - Geometric distance reduces weight but does not zero it (§17.7-grade).
  - Below-threshold pairs produce no edge.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eft_encounter_env import ExistencePressLibrary
from eft_encounter_interaction import EncounterInteraction
from eft_verified_core import DIM, Trace
from eft_trace_links import (
    TraceLinkGraph, TraceLinkEdge, compute_edge_weight, link_new_trace,
    EDGE_WEIGHT_THRESHOLD,
    W_TEMPORAL, W_GEOMETRIC, W_EIGENMODE, W_SIGNIFICANCE,
    TEMPORAL_DECAY_TAU,
)


def _trace(position, normal, significance=0.5, kernel_width=0.5,
           deposit_time=0):
    return Trace(
        position=np.array(position, dtype=float),
        significance=float(significance),
        kernel_width=float(kernel_width),
        normal=np.array(normal, dtype=float),
        deposit_time=int(deposit_time),
        current_age=0,
    )


def test_no_links_for_first_trace():
    lib = ExistencePressLibrary()
    interaction = EncounterInteraction(seed=42)
    # Drive a single admissible encounter. Use prenatal priming once
    # then deliberately deposit one trace via skin contact.
    for _ in range(5):
        interaction.process(lib.prenatal_steady())
    # After priming, possibly several traces already deposited.
    # Reset for a clean check: instantiate a fresh interaction and
    # process a single press that we know encounters.
    fresh = EncounterInteraction(seed=42)
    # Need to encounter on first press. With initial H state (mu1=0.5,
    # sigma1=0.4), early_postnatal_skin (magnitude=0.4, valence=0.7) is
    # strongly receivable.
    event = fresh.process(lib.early_postnatal_skin())
    if not event.encountered:
        pytest.skip("first-press encounter not produced under initial state")
    assert len(fresh.self_system.traces) >= 1
    assert len(fresh.trace_link_graph.edges) == 0


def test_links_form_between_temporally_close_traces():
    lib = ExistencePressLibrary()
    interaction = EncounterInteraction(seed=42)
    for _ in range(5):
        interaction.process(lib.prenatal_steady())
    interaction.process(lib.early_postnatal_skin())
    assert len(interaction.trace_link_graph.edges) >= 1
    # Most recent trace should have at least one neighbor.
    last_idx = len(interaction.self_system.traces) - 1
    neighbors = interaction.trace_link_graph.neighbors(last_idx)
    assert len(neighbors) >= 1


def test_temporal_proximity_dominates_weight():
    t_a = _trace([0., 0, 0, 0, 0], [1., 0, 0, 0, 0])
    t_b = _trace([0., 0, 0, 0, 0], [1., 0, 0, 0, 0])
    weight_close, comp_close = compute_edge_weight(t_a, t_b, time_i=0, time_j=5)
    weight_far, comp_far = compute_edge_weight(t_a, t_b, time_i=0, time_j=100)
    assert weight_close > weight_far
    expected_close = float(np.exp(-5.0 / TEMPORAL_DECAY_TAU))
    expected_far = float(np.exp(-100.0 / TEMPORAL_DECAY_TAU))
    assert abs(comp_close['temporal_proximity'] - expected_close) < 1e-9
    assert abs(comp_far['temporal_proximity'] - expected_far) < 1e-9


def test_eigenmode_misalignment_reduces_weight():
    aligned_a = _trace([0., 0, 0, 0, 0], [1., 0, 0, 0, 0])
    aligned_b = _trace([0., 0, 0, 0, 0], [1., 0, 0, 0, 0])
    misaligned_b = _trace([0., 0, 0, 0, 0], [0., 0, 0, 0, 1.])
    w_aligned, _ = compute_edge_weight(aligned_a, aligned_b, 0, 0)
    w_misaligned, _ = compute_edge_weight(aligned_a, misaligned_b, 0, 0)
    assert w_aligned > w_misaligned


def test_geometric_distance_reduces_weight_but_does_not_zero_it():
    # Same time, same normal, far apart in M^5.
    t_close_a = _trace([0., 0, 0, 0, 0], [1., 0, 0, 0, 0])
    t_close_b = _trace([0.1, 0, 0, 0, 0], [1., 0, 0, 0, 0])
    t_far_b = _trace([6., 0, 0, 0, 0], [1., 0, 0, 0, 0])
    w_close, _ = compute_edge_weight(t_close_a, t_close_b, 0, 0)
    w_far, _ = compute_edge_weight(t_close_a, t_far_b, 0, 0)
    assert w_far < w_close
    # §17.7-grade: geometry is NOT a hard suppressor. With temporal
    # proximity 1.0 (tp=1), eigenmode alignment 1.0 (ea=1), significance
    # 0.5 (sf=0.5), the floor is 0.50*1 + 0.15*1 + 0.10*0.5 = 0.70, so
    # even at extreme geometric distance the weight stays well above 0.4.
    assert w_far > 0.4, (
        f"geometry should not zero the weight: got {w_far}; "
        f"temporal+eigenmode+significance contributions guarantee a floor"
    )


def test_below_threshold_pairs_have_no_edge():
    # Pathologically misaligned: huge temporal gap, huge geometric gap,
    # opposite normals.
    t_a = _trace([0., 0, 0, 0, 0], [1., 0, 0, 0, 0], significance=0.05)
    t_b = _trace([10., 10, 10, 10, 10], [0., 0, 0, 0, 1.], significance=0.05)
    weight, _ = compute_edge_weight(t_a, t_b, 0, 200)
    assert weight < EDGE_WEIGHT_THRESHOLD
    # link_new_trace should add no edges in this configuration.
    graph = TraceLinkGraph()
    added = link_new_trace(graph, [t_a, t_b], [0, 200], new_trace_index=1)
    assert added == 0
    assert len(graph.edges) == 0
