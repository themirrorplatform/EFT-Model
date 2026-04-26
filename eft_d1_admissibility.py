"""
EFT D-1 ADMISSIBILITY VERIFICATION
Operational tests for Result D-1 Revision 2 clauses (A)(B)(C)(D).

Clause (E*) requires multi-R* (Construction 7) operationalization and
the SA6-preservation sequence monitor; not implemented in this module.
Multi-R* is conceptually required and architecturally established; what
remains is the code infrastructure to track multiple attractors and
evaluate the SA6 inequality across sequences. See handoff §17.6.

See docs/EFT_Result_D1_Encounter_Admissible_Deformation.docx
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict
from eft_verified_core import (
    DIM, GHOST_FLOOR, DEPOSIT_THRESHOLD,
    V_T, V_A, V_B, V_R, V_E, D4_INDICES, EXIT_INDEX
)

V_E_HORIZONTAL_THRESHOLD = 0.1   # |delta_x[V_E]| / |delta_x| under this
                                  # = horizontal step
V_E_PRESS_DRIVE_THRESHOLD = 0.1  # |press.direction[V_E]| over this
                                  # = press supplies V_E activation


@dataclass
class ClauseResult:
    clause: str
    verdict: str  # "PASS" | "FAIL" | "N/A"
    detail: str
    measurement: Optional[float] = None


@dataclass
class AdmissibilityReport:
    A: ClauseResult
    B: ClauseResult
    C: ClauseResult
    D: ClauseResult
    step_admissible: bool

    def summary(self) -> str:
        lines = [f"D-1 Step Admissibility: {'ADMISSIBLE' if self.step_admissible else 'INADMISSIBLE'}"]
        for c in [self.A, self.B, self.C, self.D]:
            lines.append(f"  ({c.clause}) {c.verdict}: {c.detail}")
        return "\n".join(lines)


def verify_clause_A(event) -> ClauseResult:
    """(A) Chùveld threshold: significance >= DEPOSIT_THRESHOLD."""
    if not event.encountered:
        return ClauseResult("A", "N/A",
            "Event did not encounter; clause (A) not applicable.")
    if event.self_result is None:
        return ClauseResult("A", "FAIL",
            "Event encountered but self_result is missing.")
    sig = event.self_result.get('significance')
    if sig is None:
        return ClauseResult("A", "FAIL",
            "Event encountered but significance not reported.")
    passes = sig >= DEPOSIT_THRESHOLD
    return ClauseResult(
        "A",
        "PASS" if passes else "FAIL",
        f"significance={sig:.4f} {'>=' if passes else '<'} s_deposit={DEPOSIT_THRESHOLD}",
        measurement=float(sig)
    )


def verify_clause_B(event) -> ClauseResult:
    """(B) Trace deposition: exactly one S_trace term added."""
    if not event.encountered:
        delta = event.traces_count_after - event.traces_count_before
        if delta == 0:
            return ClauseResult("B", "N/A",
                "Event did not encounter; no trace deposited (correct).")
        return ClauseResult("B", "FAIL",
            f"Event did not encounter but {delta} trace(s) deposited.")
    delta = event.traces_count_after - event.traces_count_before
    passes = (delta == 1)
    return ClauseResult(
        "B",
        "PASS" if passes else "FAIL",
        f"traces appended = {delta} (expected exactly 1)",
        measurement=float(delta)
    )


def verify_clause_C(event) -> ClauseResult:
    """(C) Sub-Riemannian horizontal reachability — operational proxy.

    This does NOT prove sub-Riemannian horizontality directly; it checks
    whether the observed R* displacement is consistent with the expected
    horizontal/bracket-escape pattern. R* displacement is one observable
    shadow of the encounter deformation; the deformation itself lives in
    g_total. A first-principles test using g_total|D^4 explicitly is
    future work; this proxy is the entry point for it.

    Predicate: the step is C-consistent if either
      - the V_E component of the R* displacement is small relative to
        the step (consistent with horizontal motion in D^4), OR
      - the external press supplied non-negligible V_E activation
        (consistent with bracket escape via external drive).
    """
    if not event.encountered:
        return ClauseResult("C", "N/A",
            "Event did not encounter; clause (C) not applicable.")
    if event.R_star_before is None or event.R_star_after_capture is None:
        return ClauseResult("C", "FAIL",
            "R_star_before/after not captured on event.")
    delta_x = event.R_star_after_capture - event.R_star_before
    delta_norm = float(np.linalg.norm(delta_x))
    if delta_norm < 1e-6:
        return ClauseResult("C", "PASS",
            "R* displacement effectively zero (vacuously consistent)",
            measurement=0.0)
    v_e_component = abs(float(delta_x[V_E]))
    v_e_fraction = v_e_component / delta_norm
    if v_e_fraction < V_E_HORIZONTAL_THRESHOLD:
        return ClauseResult(
            "C", "PASS",
            f"V_E fraction {v_e_fraction:.3f} < threshold {V_E_HORIZONTAL_THRESHOLD} "
            f"(consistent with horizontal R* motion)",
            measurement=float(v_e_fraction)
        )
    press_v_e = abs(float(event.press.direction[V_E]))
    is_external = event.enc_args.get('is_external', True) if event.enc_args else True
    if press_v_e > V_E_PRESS_DRIVE_THRESHOLD and is_external:
        return ClauseResult(
            "C", "PASS",
            f"V_E fraction {v_e_fraction:.3f} consistent with external press "
            f"V_E direction {press_v_e:.3f} (bracket-escape proxy)",
            measurement=float(v_e_fraction)
        )
    return ClauseResult(
        "C", "FAIL",
        f"V_E fraction {v_e_fraction:.3f}: R* motion not horizontal AND not "
        f"externally driven (press V_E={press_v_e:.3f}, external={is_external})",
        measurement=float(v_e_fraction)
    )


def verify_clause_D(event) -> ClauseResult:
    """(D) Ghost floor: deposited trace bounded below r* in long-time limit.

    Trace.effective_significance() enforces this at every read. The test
    simulates the long-time limit on a hypothetical trace at the event's
    reported significance and confirms the floor holds.
    """
    if not event.encountered:
        return ClauseResult("D", "N/A",
            "Event did not encounter; clause (D) not applicable.")
    from eft_verified_core import Trace
    sig = event.self_result.get('significance', 0.0) if event.self_result else 0.0
    hypo = Trace(
        position=np.zeros(DIM),
        significance=float(sig),
        kernel_width=0.5,
        normal=np.array([1.0, 0, 0, 0, 0]),
        deposit_time=0,
        current_age=10**6,
    )
    eff = hypo.effective_significance()
    passes = eff >= GHOST_FLOOR
    return ClauseResult(
        "D",
        "PASS" if passes else "FAIL",
        f"effective_significance at age 10^6 = {eff:.4f} {'>=' if passes else '<'} ghost floor {GHOST_FLOOR}",
        measurement=float(eff)
    )


def verify_admissibility(event) -> AdmissibilityReport:
    """Run clauses A, B, C, D on a single EncounterEvent. Clause E* is
    sequence-level and not run here; see module docstring."""
    A = verify_clause_A(event)
    B = verify_clause_B(event)
    C = verify_clause_C(event)
    D = verify_clause_D(event)
    step_admissible = (
        event.encountered and
        A.verdict == "PASS" and
        B.verdict == "PASS" and
        C.verdict == "PASS" and
        D.verdict == "PASS"
    )
    return AdmissibilityReport(A=A, B=B, C=C, D=D, step_admissible=step_admissible)
