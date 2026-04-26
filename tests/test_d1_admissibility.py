"""
Tests for Result D-1 Revision 2 admissibility verification.

Covers clauses (A) chùveld threshold, (B) trace deposition, (C) sub-Riemannian
horizontal reachability proxy, (D) ghost floor. Clause (E*) is sequence-level
and not implemented; see eft_d1_admissibility module docstring.
"""
import os
import sys
from dataclasses import dataclass

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eft_encounter_env import ExistencePress, ExistencePressLibrary
from eft_encounter_interaction import EncounterInteraction, EncounterEvent
from eft_d1_admissibility import (
    verify_admissibility,
    verify_clause_A,
    verify_clause_B,
    verify_clause_C,
    verify_clause_D,
)
from eft_verified_core import DIM, V_E


def _press_like(template, magnitude, label, **overrides):
    return ExistencePress(
        magnitude=magnitude,
        valence=overrides.get('valence', template.valence),
        direction=overrides.get('direction', template.direction.copy()),
        rhythm=overrides.get('rhythm', template.rhythm.copy()),
        persistence=overrides.get('persistence', template.persistence),
        label=label,
        press_class="test",
    )


def _fake_event(encountered=True, sig=0.5,
                R_before=None, R_after=None,
                traces_before=0, traces_after=1,
                press_direction=None):
    if R_before is None:
        R_before = np.zeros(DIM)
    if R_after is None:
        R_after = np.zeros(DIM)
    if press_direction is None:
        press_direction = np.array([0.5, 0.5, 0.5, 0.5, 0.0])
    press = ExistencePress(
        magnitude=0.5, valence=0.5,
        direction=press_direction,
        rhythm=np.ones(8) / 8.0,
        persistence=1.0,
        label="fake", press_class="test",
    )
    event = EncounterEvent(
        t=1, press=press, Lambda_act={}, r_scores={},
        C_before=0.0, sigma_profile=np.zeros(DIM),
        enc_args={'is_external': True},
    )
    event.encountered = encountered
    event.self_result = {'significance': sig} if encountered else None
    event.R_star_before = R_before
    event.R_star_after_capture = R_after
    event.traces_count_before = traces_before
    event.traces_count_after = traces_after
    return event


def test_admissible_encounter_passes_all_four_clauses():
    lib = ExistencePressLibrary()
    interaction = EncounterInteraction(seed=42)
    for _ in range(5):
        interaction.process(lib.prenatal_steady())
    event = interaction.process(lib.early_postnatal_skin())
    assert event.encountered is True, (
        f"setup expected encounter; got {event.no_encounter_reason!r}"
    )
    report = verify_admissibility(event)
    assert report.step_admissible is True, report.summary()
    assert report.A.verdict == "PASS"
    assert report.B.verdict == "PASS"
    assert report.C.verdict == "PASS"
    assert report.D.verdict == "PASS"


def test_no_encounter_event_clauses_are_NA():
    lib = ExistencePressLibrary()
    interaction = EncounterInteraction(seed=42)
    for _ in range(5):
        interaction.process(lib.prenatal_steady())
    # Sub-threshold press: tiny magnitude AND near-zero persistence AND
    # rhythm/direction mismatched with the primed Lambda_act, so multiple
    # receivability channels drop below theta_min and support_met fails.
    tiny = ExistencePress(
        magnitude=0.05, valence=0.0,
        direction=np.array([0.0, 0.0, 0.0, 0.0, 1.0]),
        rhythm=np.array([0., 0., 0., 0., 1., 0., 0., 0.]),
        persistence=0.01,
        label="too_small", press_class="test",
    )
    event = interaction.process(tiny)
    assert event.encountered is False, (
        f"setup expected sub-threshold non-encounter; "
        f"got C_before={event.C_before:.3f} sigma={event.r_scores.get('sigma_profile')}"
    )
    report = verify_admissibility(event)
    assert report.step_admissible is False
    for c in (report.A, report.B, report.C, report.D):
        assert c.verdict in ("N/A", "PASS"), (
            f"clause {c.clause} verdict={c.verdict} on no-encounter event: {c.detail}"
        )


def test_overwhelm_press_is_inadmissible():
    lib = ExistencePressLibrary()
    interaction = EncounterInteraction(seed=42)
    for _ in range(5):
        interaction.process(lib.prenatal_steady())
    huge = _press_like(lib.early_postnatal_skin(), magnitude=2.0,
                       label="overwhelm")
    event = interaction.process(huge)
    assert event.encountered is False
    assert "magnitude_overwhelm" in event.no_encounter_reason
    report = verify_admissibility(event)
    assert report.step_admissible is False
    for c in (report.A, report.B, report.C, report.D):
        assert c.verdict == "N/A", (
            f"overwhelm event should yield N/A for all clauses; "
            f"clause {c.clause}={c.verdict} ({c.detail})"
        )


def test_clause_A_fails_below_threshold():
    event = _fake_event(encountered=True, sig=0.05)
    result = verify_clause_A(event)
    assert result.verdict == "FAIL"
    from eft_verified_core import DEPOSIT_THRESHOLD
    assert result.measurement is not None and result.measurement < DEPOSIT_THRESHOLD


def test_clause_B_fails_when_trace_count_unchanged():
    event = _fake_event(encountered=True, traces_before=3, traces_after=3)
    result = verify_clause_B(event)
    assert result.verdict == "FAIL"


def test_clause_C_passes_for_horizontal_step():
    event = _fake_event(
        encountered=True,
        R_before=np.zeros(DIM),
        R_after=np.array([0.3, 0.2, 0.1, 0.1, 0.0]),
    )
    result = verify_clause_C(event)
    assert result.verdict == "PASS", result.detail


def test_clause_C_fails_for_internal_v_e_drift():
    # Pure V_E displacement, press supplies no V_E direction.
    event = _fake_event(
        encountered=True,
        R_before=np.zeros(DIM),
        R_after=np.array([0.0, 0.0, 0.0, 0.0, 0.5]),
        press_direction=np.array([0.5, 0.5, 0.5, 0.5, 0.0]),
    )
    result = verify_clause_C(event)
    assert result.verdict == "FAIL", result.detail
    assert result.measurement is not None
    assert abs(result.measurement - 1.0) < 0.01


def test_clause_D_holds_for_any_significance():
    for sig in (0.01, 0.5, 5.0):
        event = _fake_event(encountered=True, sig=sig)
        result = verify_clause_D(event)
        assert result.verdict == "PASS", (
            f"sig={sig}: {result.detail}"
        )
