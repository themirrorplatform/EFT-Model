"""
EFT Trace Link Graph — Level 3 of the staged outputs(T) ladder.

Edges between traces, weighted by:
  1. temporal proximity (primary)
  2. geometric distance in M^5
  3. eigenmode direction alignment
  4. significance strength
  5. co-activation/replay history (FUTURE — Phase 2.3, marked as
     extension point below)

The four implemented factors combine ADDITIVELY with priority-weighted
coefficients summing to 1.0:

    weight = 0.50 * tp + 0.25 * gp + 0.15 * ea + 0.10 * sf

This makes temporal proximity primary (50% of weight) without making
any factor a hard multiplicative gate. Two temporally close traces at
geometric distance still link; two geometrically close traces at
temporal distance still link weakly. This matches engram literature:
linking is primarily temporal contiguity but later reactivation can
also create links between geometrically/semantically similar traces.

Cluster detection over this graph is Phase 2.2 (next session). This
module only constructs and queries the link graph.

Architectural boundary preserved verbatim: Level 5 must be cluster-replay-
as-press, not generative emission. Cluster detection here produces only
edges; it never declares attractors. Promotion of a cluster to a confirmed
secondary R* requires self-encounter and SA6 coupling at Level 7.

See RECONNAISSANCE_UPDATES.md for the full ladder.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from eft_verified_core import DIM, Trace

# Edge-weight hyperparameters. These are calibration-relevant; revisit
# after Phase 2.2 cluster detection runs against synthetic-DEAP data.
TEMPORAL_DECAY_TAU = 20.0       # encounters; exponential decay scale
GEOMETRIC_SIGMA = 1.5            # M^5 distance scale

# Priority-weighted coefficients for the additive edge-weight formula.
# These sum to 1.0 by construction — temporal proximity dominates at
# 50%, geometric and eigenmode/significance modulate.
W_TEMPORAL = 0.50
W_GEOMETRIC = 0.25
W_EIGENMODE = 0.15
W_SIGNIFICANCE = 0.10

EDGE_WEIGHT_THRESHOLD = 0.15     # below this, no edge is created


@dataclass
class TraceLinkEdge:
    i: int          # index into the trace store
    j: int
    weight: float
    components: Dict[str, float]  # named contributions for diagnostics


@dataclass
class TraceLinkGraph:
    """Graph over trace indices with weighted edges. Sparse: not every
    pair of traces has an edge. New edges are added as new traces
    deposit; existing edges are not recomputed (their weights are
    fixed at creation time, since temporal proximity is decided at
    that moment).
    """
    edges: List[TraceLinkEdge] = field(default_factory=list)
    _edges_by_node: Dict[int, List[int]] = field(default_factory=dict)

    def neighbors(self, i: int) -> List[Tuple[int, float]]:
        """Return [(neighbor_index, edge_weight), ...] for trace i."""
        edge_indices = self._edges_by_node.get(i, [])
        out = []
        for ei in edge_indices:
            e = self.edges[ei]
            other = e.j if e.i == i else e.i
            out.append((other, e.weight))
        return out

    def degree(self, i: int) -> int:
        return len(self._edges_by_node.get(i, []))

    def add_edge(self, edge: TraceLinkEdge) -> None:
        edge_idx = len(self.edges)
        self.edges.append(edge)
        self._edges_by_node.setdefault(edge.i, []).append(edge_idx)
        self._edges_by_node.setdefault(edge.j, []).append(edge_idx)


def _temporal_proximity(t_i: int, t_j: int) -> float:
    """Smooth exponential in temporal distance. Higher = closer in time."""
    return float(np.exp(-abs(t_i - t_j) / TEMPORAL_DECAY_TAU))


def _geometric_proximity(x_i: np.ndarray, x_j: np.ndarray) -> float:
    """Gaussian in M^5 distance. Higher = closer in space."""
    d = float(np.linalg.norm(x_i - x_j))
    return float(np.exp(-d * d / (2 * GEOMETRIC_SIGMA * GEOMETRIC_SIGMA)))


def _eigenmode_alignment(n_i: np.ndarray, n_j: np.ndarray) -> float:
    """Cosine similarity, clipped to [0, 1] (negative alignment = no link)."""
    ni = float(np.linalg.norm(n_i))
    nj = float(np.linalg.norm(n_j))
    if ni < 1e-10 or nj < 1e-10:
        return 0.0
    cos = float(np.dot(n_i, n_j)) / (ni * nj)
    return max(0.0, cos)


def _significance_factor(s_i: float, s_j: float) -> float:
    """Geometric mean of effective significances. Both inputs are >=
    GHOST_FLOOR (0.05) by Trace.effective_significance, so this is
    bounded above 0.05."""
    val = float(np.sqrt(max(s_i, 0.0) * max(s_j, 0.0)))
    # Clip to [0, 1] for combining with the other factors which are
    # also in [0, 1].
    return min(val, 1.0)


def compute_edge_weight(trace_i: Trace, trace_j: Trace,
                        time_i: int, time_j: int) -> Tuple[float, Dict[str, float]]:
    """Combine the four implemented weight components additively with
    priority-weighted coefficients summing to 1.0.

    EXTENSION POINT (Phase 2.3): Add a fifth component for
    co-activation/replay history once cluster reactivation is
    implemented. Add it as W_COACTIVATION * coactivation_score and
    adjust the other coefficients to keep the sum at 1.0.
    """
    tp = _temporal_proximity(time_i, time_j)
    gp = _geometric_proximity(trace_i.position, trace_j.position)
    ea = _eigenmode_alignment(trace_i.normal, trace_j.normal)
    sf = _significance_factor(trace_i.effective_significance(),
                              trace_j.effective_significance())
    weight = (W_TEMPORAL * tp + W_GEOMETRIC * gp +
              W_EIGENMODE * ea + W_SIGNIFICANCE * sf)
    components = {
        'temporal_proximity': tp,
        'geometric_proximity': gp,
        'eigenmode_alignment': ea,
        'significance_factor': sf,
    }
    return float(weight), components


def link_new_trace(graph: TraceLinkGraph,
                   traces: List[Trace],
                   trace_times: List[int],
                   new_trace_index: int) -> int:
    """Add edges between the newly-deposited trace and existing traces
    where the edge weight exceeds threshold. Returns the number of
    edges added."""
    new_trace = traces[new_trace_index]
    new_time = trace_times[new_trace_index]
    added = 0
    for j in range(len(traces)):
        if j == new_trace_index:
            continue
        weight, components = compute_edge_weight(
            new_trace, traces[j], new_time, trace_times[j]
        )
        if weight >= EDGE_WEIGHT_THRESHOLD:
            edge = TraceLinkEdge(
                i=new_trace_index, j=j,
                weight=weight, components=components
            )
            graph.add_edge(edge)
            added += 1
    return added
