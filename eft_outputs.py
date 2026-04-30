"""
EFT Outputs — Level 1 of the staged outputs(T) ladder.

An OutputRecord is the system's structured response-to-press, reified
from EFTSystem.encounter(). It is NOT yet a thought, internal emission,
or generative product — those are higher levels of the ladder (Level 5,
cluster-replay-as-press; not implemented in this phase).

Architectural boundary preserved verbatim: Level 5 must be cluster-replay-
as-press, not generative emission. The system reactivates a cluster and
reintroduces it as an internally-sourced press into encounter(). This is
what distinguishes EFT from a chatbot pretending to think.

See RECONNAISSANCE_UPDATES.md for the full ladder.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from eft_verified_core import DIM


@dataclass
class OutputRecord:
    """Level 1 output: the system's structured response to a single
    encounter event. Carries the system's active contribution
    (interpretation, mode, R* movement, dominant eigenmode shape,
    residue) — not just what happened to it.

    Naming note (§17.7 discipline): response_position is the post-encounter
    R* position, i.e., where the system's response settled. It is NOT
    the encounter landing position x_enc (which is internal to
    EFTSystem.encounter and not exposed). Anyone reading this field
    should understand it as the attractor-after-integration, not the
    press-impact-point.
    """
    t: int                                  # encounter timestamp
    source_press_label: str                 # which press generated this
    response_mode: str                      # generative/regenerative/neutral/degenerate
    significance: float                     # encounter significance
    response_position: np.ndarray           # post-encounter R* in M^5
    R_star_before: np.ndarray               # R* before this encounter
    R_star_after: np.ndarray                # R* after this encounter
                                            # (== response_position; kept
                                            # as separate field for
                                            # clarity and future use)
    dominant_eigenmode_shape: np.ndarray    # eigenmode_weights normalized
    uncertainty_residue: float              # 1 - max(|n_p|), captures
                                            # how undetermined the
                                            # eigenmode shape was
    is_externally_caused: bool              # True if from external press,
                                            # False if from self-encounter
                                            # (Level 6 — not yet active)
    trace_count_after: int                  # link to trace substrate


@dataclass
class OutputHistory:
    """Bounded history of OutputRecords. Capped to prevent unbounded
    memory growth. Older outputs roll off; this is a working window,
    not a permanent archive."""
    records: List[OutputRecord] = field(default_factory=list)
    max_size: int = 500

    def append(self, record: OutputRecord) -> None:
        self.records.append(record)
        if len(self.records) > self.max_size:
            self.records.pop(0)

    def __len__(self) -> int:
        return len(self.records)

    def recent(self, n: int) -> List[OutputRecord]:
        return self.records[-n:]


def output_record_from_event(event) -> Optional[OutputRecord]:
    """Construct an OutputRecord from an EncounterEvent. Returns None
    if the event did not encounter (no output to record)."""
    if not event.encountered or event.self_result is None:
        return None
    sr = event.self_result
    eigenmode_impact = sr.get('eigenmode_impact', {})
    if isinstance(eigenmode_impact, dict):
        # eigenmode_impact comes as a dict keyed by eigenmode name in the
        # current EFTSystem report; convert to a fixed-order array.
        from eft_verified_core import V_T, V_A, V_B, V_R, V_E
        names = ['temporal', 'agency', 'boundary', 'reality', 'exit']
        eigenmode_shape = np.array([float(eigenmode_impact.get(n, 0.0)) for n in names])
    else:
        eigenmode_shape = np.array(eigenmode_impact, dtype=float)
    norm = float(np.linalg.norm(eigenmode_shape))
    if norm > 1e-10:
        eigenmode_shape = eigenmode_shape / norm
    uncertainty = 1.0 - float(np.max(np.abs(eigenmode_shape))) if norm > 1e-10 else 1.0
    is_external = event.enc_args.get('is_external', True) if event.enc_args else True
    response_pos = np.array(sr.get('R_star', np.zeros(DIM))).copy()
    return OutputRecord(
        t=event.t,
        source_press_label=event.press.label,
        response_mode=sr.get('mode', 'unknown'),
        significance=float(sr.get('significance', 0.0)),
        response_position=response_pos,
        R_star_before=event.R_star_before.copy() if event.R_star_before is not None else np.zeros(DIM),
        R_star_after=event.R_star_after_capture.copy() if event.R_star_after_capture is not None else np.zeros(DIM),
        dominant_eigenmode_shape=eigenmode_shape,
        uncertainty_residue=uncertainty,
        is_externally_caused=is_external,
        trace_count_after=event.traces_count_after,
    )
