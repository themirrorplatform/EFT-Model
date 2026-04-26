"""
EFT Clusters — Level 4 of the staged outputs(T) ladder.

Detects candidate secondary attractors over the trace link graph.
Produces CandidateSecondaryAttractor objects representing connected,
coherent, distinct clusters of traces.

CANDIDATES ARE NOT SECONDARY R*s. A candidate is an observation
about cluster structure in the trace cloud. Promotion to recognized
secondary R* happens at Level 7 (Phase 2.4) after self-encounter
and SA6 coupling validate the candidate. This module must not
introduce any concept of a "recognized" or "active" secondary R*.

Cluster method: connected components over the link graph, filtered
by size, internal coherence, and distinction from primary R*.
Soft distinction criterion: spatial distance OR orientation
distinction qualifies. See architecture decision A in Phase 2.2
prompt.

See RECONNAISSANCE_UPDATES.md for the full ladder.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple, Dict
from eft_verified_core import DIM, Trace, compute_Hessian_at_R_star
from eft_trace_links import TraceLinkGraph

# Cluster validity thresholds. PROVISIONAL — calibrated against
# architectural commitment (secondary R*s are significant). Revisit
# if synthetic-DEAP behavior surfaces tuning evidence.
MIN_CLUSTER_SIZE = 3
MIN_MEAN_EDGE_WEIGHT = 0.30

# Soft distinction parameters.
BASIN_RADIUS_FACTOR = 0.75            # candidate centroid must be at
                                       # least this fraction of primary
                                       # basin radius away to qualify
                                       # via the spatial route
ORIENTATION_DISTINCTION_THRESHOLD = 0.70  # cosine similarity below this
                                           # qualifies via the
                                           # orientation route (i.e.,
                                           # angle > ~45 degrees)

# Fallback when primary Hessian isn't positive-definite.
MIN_BASIN_RADIUS_FALLBACK = 0.5
MAX_BASIN_RADIUS_CAP = 5.0


@dataclass
class CandidateSecondaryAttractor:
    """A candidate for secondary attractor status. NOT a recognized
    secondary R*. Records the trace cluster's geometry, coherence,
    and qualifying route (spatial vs orientation distinction).
    """
    member_indices: List[int]              # indices into the trace store
    centroid: np.ndarray                   # mean position in M^5
    orientation: np.ndarray                # mean normal direction (unit)
    size: int                              # = len(member_indices)
    mean_edge_weight: float                # internal coherence
    distance_to_primary_R_star: float      # in M^5 metric (Euclidean here)
    orientation_cosine_to_primary: float   # in [-1, 1]
    qualifying_route: str                  # "spatial" | "orientation" | "both"
    formation_timestamp: int               # when this candidate was detected


def _connected_components(graph: TraceLinkGraph, n_traces: int) -> List[Set[int]]:
    """Standard connected-components over the link graph, treating
    every edge above EDGE_WEIGHT_THRESHOLD as a connection. Returns
    list of node-index sets; isolated nodes are excluded.
    """
    visited: Set[int] = set()
    components: List[Set[int]] = []
    for start in range(n_traces):
        if start in visited:
            continue
        neighbors = graph.neighbors(start)
        if not neighbors:
            # isolated node, skip
            visited.add(start)
            continue
        # BFS
        component: Set[int] = {start}
        queue = [start]
        visited.add(start)
        while queue:
            node = queue.pop()
            for (other, _w) in graph.neighbors(node):
                if other not in visited:
                    visited.add(other)
                    component.add(other)
                    queue.append(other)
        if len(component) >= MIN_CLUSTER_SIZE:
            components.append(component)
    return components


def _compute_mean_edge_weight(component: Set[int],
                                graph: TraceLinkGraph) -> float:
    """Mean weight over edges fully contained in this component."""
    total = 0.0
    count = 0
    for node in component:
        for (other, w) in graph.neighbors(node):
            if other in component and other > node:  # avoid double-counting
                total += w
                count += 1
    if count == 0:
        return 0.0
    return total / count


def _compute_cluster_geometry(component: Set[int],
                                traces: List[Trace]) -> Tuple[np.ndarray, np.ndarray]:
    """Cluster centroid (significance-weighted mean position) and
    orientation (significance-weighted mean unit normal).
    """
    indices = list(component)
    positions = np.array([traces[i].position for i in indices])
    normals = np.array([traces[i].normal for i in indices])
    weights = np.array([traces[i].effective_significance() for i in indices])
    w_norm = weights / (weights.sum() + 1e-12)
    centroid = (w_norm[:, None] * positions).sum(axis=0)
    orientation = (w_norm[:, None] * normals).sum(axis=0)
    o_norm = float(np.linalg.norm(orientation))
    if o_norm > 1e-10:
        orientation = orientation / o_norm
    else:
        # degenerate case: orientations cancel out; use first member
        orientation = traces[indices[0]].normal.copy()
        o_norm2 = float(np.linalg.norm(orientation))
        if o_norm2 > 1e-10:
            orientation = orientation / o_norm2
    return centroid, orientation


def _estimate_primary_basin_radius(R_star_primary: np.ndarray,
                                    traces: List[Trace],
                                    G_field: np.ndarray,
                                    G_body: np.ndarray) -> float:
    """Estimate basin radius from Hessian eigenvalues at primary R*.
    Falls back to trace-cloud scale if Hessian isn't positive-definite.
    """
    try:
        H = compute_Hessian_at_R_star(R_star_primary, traces, G_field, G_body)
        eigvals = np.linalg.eigvalsh(H)
        # Use smallest positive eigenvalue (weakest direction defines
        # basin extent).
        positive = eigvals[eigvals > 1e-6]
        if len(positive) > 0:
            radius = 1.0 / float(np.sqrt(positive.min()))
            # Cap to avoid pathological values
            return float(np.clip(radius, MIN_BASIN_RADIUS_FALLBACK,
                                 MAX_BASIN_RADIUS_CAP))
    except Exception:
        pass
    # Fallback: trace-cloud geometric scale
    if len(traces) < 2:
        return MIN_BASIN_RADIUS_FALLBACK
    positions = np.array([t.position for t in traces])
    mean_pos = positions.mean(axis=0)
    mean_dist = float(np.linalg.norm(positions - mean_pos, axis=1).mean())
    return float(np.clip(mean_dist, MIN_BASIN_RADIUS_FALLBACK,
                         MAX_BASIN_RADIUS_CAP))


def _primary_orientation(R_star_primary: np.ndarray,
                          traces: List[Trace],
                          G_field: np.ndarray,
                          G_body: np.ndarray) -> np.ndarray:
    """Dominant eigenvector of the Hessian at primary R* — the
    direction primary R* is most strongly attractive in. Falls back
    to first standard basis vector if Hessian is degenerate.
    """
    try:
        H = compute_Hessian_at_R_star(R_star_primary, traces, G_field, G_body)
        eigvals, eigvecs = np.linalg.eigh(H)
        # Largest absolute eigenvalue's eigenvector
        idx = int(np.argmax(np.abs(eigvals)))
        v = eigvecs[:, idx]
        n = float(np.linalg.norm(v))
        if n > 1e-10:
            return v / n
    except Exception:
        pass
    return np.eye(DIM)[0]


def detect_candidate_secondary_attractors(
    traces: List[Trace],
    graph: TraceLinkGraph,
    R_star_primary: np.ndarray,
    G_field: np.ndarray,
    G_body: np.ndarray,
    current_time: int,
) -> List[CandidateSecondaryAttractor]:
    """Run cluster detection. Returns list of CandidateSecondaryAttractor
    objects, one per qualifying cluster. Empty list if no clusters
    qualify.

    This function does NOT mutate any state. It is a pure observation
    over the current trace cloud and link graph.

    Calibration note: the thresholds (MIN_CLUSTER_SIZE=3,
    MIN_MEAN_EDGE_WEIGHT=0.30, BASIN_RADIUS_FACTOR=0.75,
    ORIENTATION_DISTINCTION_THRESHOLD=0.70) are PROVISIONAL. If
    synthetic-DEAP at seed=42 over 30 trials produces more than 3
    candidates, the finding is recorded in RECONNAISSANCE_UPDATES.md
    as a diagnostic — NOT used to retune thresholds. Three hypotheses
    to consider in that case: synthetic generator artifact, link-graph
    permissiveness, or need for the cross-encounter consolidation gate.
    """
    n_traces = len(traces)
    if n_traces < MIN_CLUSTER_SIZE:
        return []

    components = _connected_components(graph, n_traces)
    if not components:
        return []

    basin_radius = _estimate_primary_basin_radius(
        R_star_primary, traces, G_field, G_body
    )
    primary_orient = _primary_orientation(
        R_star_primary, traces, G_field, G_body
    )

    candidates = []
    for component in components:
        mean_w = _compute_mean_edge_weight(component, graph)
        if mean_w < MIN_MEAN_EDGE_WEIGHT:
            continue
        centroid, orientation = _compute_cluster_geometry(component, traces)
        dist_to_primary = float(np.linalg.norm(centroid - R_star_primary))
        orient_cos = float(np.dot(orientation, primary_orient))
        # clip for numerical safety
        orient_cos = float(np.clip(orient_cos, -1.0, 1.0))

        spatial_qualifies = dist_to_primary >= BASIN_RADIUS_FACTOR * basin_radius
        orientation_qualifies = orient_cos < ORIENTATION_DISTINCTION_THRESHOLD

        if spatial_qualifies and orientation_qualifies:
            route = "both"
        elif spatial_qualifies:
            route = "spatial"
        elif orientation_qualifies:
            route = "orientation"
        else:
            # Cluster exists, has internal coherence, but is not
            # meaningfully distinct from primary R*. Not a candidate.
            continue

        candidates.append(CandidateSecondaryAttractor(
            member_indices=sorted(component),
            centroid=centroid,
            orientation=orientation,
            size=len(component),
            mean_edge_weight=mean_w,
            distance_to_primary_R_star=dist_to_primary,
            orientation_cosine_to_primary=orient_cos,
            qualifying_route=route,
            formation_timestamp=current_time,
        ))
    return candidates
