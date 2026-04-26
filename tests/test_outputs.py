"""
Tests for Phase 2.1 Level 1 — OutputRecord and OutputHistory.

OutputRecord is the structured response to a single encounter event.
OutputHistory is the bounded working window over recent outputs. These
tests cover construction from events, the cap mechanic, ordering under
normal flow, and absence of outputs on non-encounters / overwhelm.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eft_encounter_env import ExistencePress, ExistencePressLibrary
from eft_encounter_interaction import EncounterInteraction
from eft_outputs import OutputHistory, OutputRecord, output_record_from_event
from eft_verified_core import DEPOSIT_THRESHOLD, DIM


def test_output_record_constructed_from_admissible_encounter():
    lib = ExistencePressLibrary()
    interaction = EncounterInteraction(seed=42)
    for _ in range(5):
        interaction.process(lib.prenatal_steady())
    event = interaction.process(lib.early_postnatal_skin())
    assert event.encountered, f"setup expected encounter, got {event.no_encounter_reason!r}"
    assert len(interaction.output_history) >= 1
    last = interaction.output_history.records[-1]
    assert last.source_press_label == "early_postnatal_skin"
    assert last.significance > DEPOSIT_THRESHOLD
    assert last.response_mode in ("generative", "regenerative", "neutral", "degenerate")
    assert last.is_externally_caused is True


def test_output_history_cap_unit():
    history = OutputHistory(max_size=10)
    for i in range(25):
        rec = OutputRecord(
            t=i, source_press_label=f"p{i}", response_mode="generative",
            significance=0.5, response_position=np.zeros(DIM),
            R_star_before=np.zeros(DIM), R_star_after=np.zeros(DIM),
            dominant_eigenmode_shape=np.zeros(DIM),
            uncertainty_residue=0.0, is_externally_caused=True,
            trace_count_after=i,
        )
        history.append(rec)
    assert len(history) == 10
    # First 15 (i=0..14) rolled off; oldest remaining is i=15.
    assert history.records[0].t == 15
    assert history.records[-1].t == 24


def test_output_history_integration_no_cap_under_normal_flow():
    lib = ExistencePressLibrary()
    interaction = EncounterInteraction(seed=42)
    for _ in range(30):
        interaction.process(lib.prenatal_steady())
    assert len(interaction.output_history) <= 30
    times = [r.t for r in interaction.output_history.records]
    assert times == sorted(times), "output_history should be in temporal order"


def test_no_output_for_no_encounter_event():
    interaction = EncounterInteraction(seed=42)
    lib = ExistencePressLibrary()
    for _ in range(5):
        interaction.process(lib.prenatal_steady())
    pre_count = len(interaction.output_history)
    tiny = ExistencePress(
        magnitude=0.05, valence=0.0,
        direction=np.array([0.0, 0.0, 0.0, 0.0, 1.0]),
        rhythm=np.array([0., 0., 0., 0., 1., 0., 0., 0.]),
        persistence=0.01,
        label="too_small", press_class="test",
    )
    event = interaction.process(tiny)
    assert event.encountered is False
    assert len(interaction.output_history) == pre_count


def test_no_output_for_overwhelm_event():
    interaction = EncounterInteraction(seed=42)
    huge = ExistencePress(
        magnitude=2.0, valence=0.0,
        direction=np.array([0.2, 0.2, 0.2, 0.2, 0.9]),
        rhythm=np.ones(8) / 8.0, persistence=1.0,
        label="overwhelm", press_class="test",
    )
    event = interaction.process(huge)
    assert event.encountered is False
    assert "magnitude_overwhelm" in event.no_encounter_reason
    assert len(interaction.output_history) == 0
