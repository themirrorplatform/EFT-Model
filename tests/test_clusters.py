"""
Tests for Phase 2.2 Level 4 — CandidateSecondaryAttractor detection.

A candidate is NOT a recognized secondary R*. Tests cover:
  - Insufficient traces / no links → no candidates.
  - Coherent, distinct cluster → exactly one candidate.
  - Cluster co-located AND aligned with primary R* → no candidate.
  - Soft distinction: orientation alone qualifies; spatial alone qualifies.
  - Low-coherence cluster → no candidate.
  - Synthetic-DEAP calibration check (diagnostic, never fails).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eft_encounter_env import ExistencePressLibrary
from eft_encounter_interaction import EncounterInteraction
from eft_verified_core import DIM, Trace
from eft_trace_links import TraceLinkGraph, TraceLinkEdge
from eft_clusters import (
    detect_candidate_secondary_attractors,
    MIN_CLUSTER_SIZE, MIN_MEAN_EDGE_WEIGHT,
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


def _fully_connected_graph(n, weight, components_offset=0):
    """Build a TraceLinkGraph with n nodes (indices 0..n-1) all
    pairwise-connected at the given edge weight."""
    g = TraceLinkGraph()
    for i in range(n):
        for j in range(i + 1, n):
            g.add_edge(TraceLinkEdge(
                i=i + components_offset, j=j + components_offset,
                weight=weight,
                components={'temporal_proximity': weight,
                            'geometric_proximity': weight,
                            'eigenmode_alignment': weight,
                            'significance_factor': weight},
            ))
    return g


def _identity_G():
    return np.eye(DIM)


def test_no_candidates_with_too_few_traces():
    interaction = EncounterInteraction(seed=42)
    lib = ExistencePressLibrary()
    interaction.process(lib.early_postnatal_skin())  # likely deposits 1
    interaction.process(lib.prenatal_steady())       # likely deposits 1
    # 2 traces is below MIN_CLUSTER_SIZE=3
    assert len(interaction.self_system.traces) < MIN_CLUSTER_SIZE or \
        len(interaction.detect_candidates()) == 0
    candidates = interaction.detect_candidates()
    assert isinstance(candidates, list)
    if len(interaction.self_system.traces) < MIN_CLUSTER_SIZE:
        assert candidates == []


def test_no_candidates_when_no_links_form():
    traces = [
        _trace([0., 0, 0, 0, 0], [1., 0, 0, 0, 0]),
        _trace([1., 0, 0, 0, 0], [1., 0, 0, 0, 0]),
        _trace([2., 0, 0, 0, 0], [1., 0, 0, 0, 0]),
    ]
    empty_graph = TraceLinkGraph()
    candidates = detect_candidate_secondary_attractors(
        traces=traces, graph=empty_graph,
        R_star_primary=np.zeros(DIM),
        G_field=_identity_G(), G_body=_identity_G(),
        current_time=10,
    )
    assert candidates == []


def test_candidate_emerges_from_coherent_cluster():
    # 4 traces clustered around [2, 0, 0, 0, 0], all oriented along V_E.
    cluster_pos = np.array([2.0, 0, 0, 0, 0])
    traces = [
        _trace(cluster_pos + 0.05 * np.random.RandomState(s).randn(DIM),
               [0., 0, 0, 0, 1.])
        for s in range(4)
    ]
    graph = _fully_connected_graph(4, weight=0.6)
    candidates = detect_candidate_secondary_attractors(
        traces=traces, graph=graph,
        R_star_primary=np.zeros(DIM),
        G_field=_identity_G(), G_body=_identity_G(),
        current_time=10,
    )
    assert len(candidates) == 1
    c = candidates[0]
    assert c.size == 4
    assert c.qualifying_route in ("spatial", "orientation", "both")


def test_cluster_at_primary_R_star_with_aligned_orientation_does_not_qualify():
    # Traces co-located with primary R* (origin) and aligned with the
    # primary orientation that the Hessian-fallback returns (eye(DIM)[0]
    # i.e., V_T). Expectation: cluster has size and coherence but is
    # neither spatially nor orientationally distinct.
    traces = [
        _trace([0., 0, 0, 0, 0], [1., 0, 0, 0, 0])
        for _ in range(4)
    ]
    graph = _fully_connected_graph(4, weight=0.6)
    candidates = detect_candidate_secondary_attractors(
        traces=traces, graph=graph,
        R_star_primary=np.zeros(DIM),
        G_field=_identity_G(), G_body=_identity_G(),
        current_time=10,
    )
    assert candidates == [], (
        f"co-located + aligned cluster should not qualify; got {candidates}"
    )


def test_orientation_distinction_alone_qualifies_candidate():
    # 4 traces NEAR origin (within basin) but oriented orthogonally to
    # the primary orientation (V_T fallback). They point along V_E.
    traces = [
        _trace([0.05 * i, 0, 0, 0, 0], [0., 0, 0, 0, 1.])
        for i in range(4)
    ]
    graph = _fully_connected_graph(4, weight=0.6)
    candidates = detect_candidate_secondary_attractors(
        traces=traces, graph=graph,
        R_star_primary=np.zeros(DIM),
        G_field=_identity_G(), G_body=_identity_G(),
        current_time=10,
    )
    assert len(candidates) == 1
    assert candidates[0].qualifying_route == "orientation", (
        f"expected orientation-only route; got {candidates[0].qualifying_route} "
        f"(dist={candidates[0].distance_to_primary_R_star:.3f}, "
        f"cos={candidates[0].orientation_cosine_to_primary:.3f})"
    )


def test_spatial_distinction_alone_qualifies_candidate():
    # 4 traces FAR from primary R* (well beyond basin radius) with
    # normals aligned to primary orientation (V_T).
    traces = [
        _trace([6.0 + 0.05 * i, 0, 0, 0, 0], [1., 0, 0, 0, 0])
        for i in range(4)
    ]
    graph = _fully_connected_graph(4, weight=0.6)
    candidates = detect_candidate_secondary_attractors(
        traces=traces, graph=graph,
        R_star_primary=np.zeros(DIM),
        G_field=_identity_G(), G_body=_identity_G(),
        current_time=10,
    )
    assert len(candidates) == 1
    assert candidates[0].qualifying_route == "spatial", (
        f"expected spatial-only route; got {candidates[0].qualifying_route} "
        f"(dist={candidates[0].distance_to_primary_R_star:.3f}, "
        f"cos={candidates[0].orientation_cosine_to_primary:.3f})"
    )


def test_low_coherence_cluster_does_not_qualify():
    # 4 traces fully connected but at edge weights barely above link
    # threshold (0.16-0.18). Mean ~ 0.17, below MIN_MEAN_EDGE_WEIGHT=0.30.
    traces = [
        _trace([2.0 + 0.05 * i, 0, 0, 0, 0], [0., 0, 0, 0, 1.])
        for i in range(4)
    ]
    graph = _fully_connected_graph(4, weight=0.17)
    candidates = detect_candidate_secondary_attractors(
        traces=traces, graph=graph,
        R_star_primary=np.zeros(DIM),
        G_field=_identity_G(), G_body=_identity_G(),
        current_time=10,
    )
    assert candidates == [], (
        f"low-coherence cluster (mean weight 0.17) should not qualify; "
        f"got {candidates}"
    )


def test_synthetic_DEAP_calibration_check():
    """Diagnostic test: never fails. Prints the candidate structure
    produced by 30 synthetic-DEAP trials at seed=42."""
    from eft_deap_translator import (
        generate_synthetic_participant, participant_to_press_stream,
    )
    data = generate_synthetic_participant(n_trials=10, seed=42)
    presses = participant_to_press_stream(data, 1)
    # generate_synthetic_participant produces 10 trials max in the
    # generator's emotional_sequence; pad to 30 by re-running with
    # different seeds.
    if len(presses) < 30:
        for extra_seed in (43, 44):
            extra = generate_synthetic_participant(n_trials=10, seed=extra_seed)
            presses.extend(participant_to_press_stream(extra, extra_seed))
    presses = presses[:30]
    interaction = EncounterInteraction(seed=42)
    for p in presses:
        interaction.process(p)
    candidates = interaction.detect_candidates()
    assert isinstance(candidates, list)
    print(f"\n[DEAP calibration] {len(candidates)} candidate(s) over 30 trials")
    for i, c in enumerate(candidates):
        print(f"  candidate {i}: size={c.size}, route={c.qualifying_route}, "
              f"dist={c.distance_to_primary_R_star:.3f}, "
              f"cos={c.orientation_cosine_to_primary:.3f}, "
              f"mean_w={c.mean_edge_weight:.3f}")
