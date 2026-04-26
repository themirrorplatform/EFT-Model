"""
Tests for handoff §17.3 — R1 overwhelm ceiling at magnitude > 1.5.

The architectural rule: any press with magnitude > 1.5 must not produce an
encounter. The press still exists and is recorded; the encounter is blocked
with a visible reason "magnitude_overwhelm".

The boundary is strictly >: magnitude == 1.5 is allowed; magnitude > 1.5 is not.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eft_encounter_env import ExistencePress, ExistencePressLibrary
from eft_encounter_interaction import EncounterInteraction


def _press_like(template, magnitude, label):
    """Reuse a library press's shape (direction, rhythm, valence, persistence)
    with a different magnitude. Lets us test the magnitude axis in isolation
    against shapes already known to be receivable under matching priming."""
    return ExistencePress(
        magnitude=magnitude,
        valence=template.valence,
        direction=template.direction.copy(),
        rhythm=template.rhythm.copy(),
        persistence=template.persistence,
        label=label,
        press_class="test",
    )


def test_press_magnitude_above_ceiling_does_not_encounter():
    """A press with magnitude=2.0 must NOT encounter regardless of priming.
    The §17.3 ceiling fires before the gate is even consulted."""
    lib = ExistencePressLibrary()
    interaction = EncounterInteraction(seed=42)
    # Standard prenatal priming (matching the existing phase test pattern).
    for _ in range(15):
        interaction.process(lib.prenatal_steady())

    press = _press_like(lib.early_postnatal_skin(), magnitude=2.0,
                        label="overwhelm_test")
    event = interaction.process(press)

    assert event.encountered is False
    assert "magnitude_overwhelm" in event.no_encounter_reason
    assert "2.000" in event.no_encounter_reason


def test_press_magnitude_at_ceiling_can_encounter():
    """A press with magnitude=1.5 (exactly at the boundary) must be allowed
    to encounter under appropriate priming. Confirms the check is strictly
    >, not >=."""
    lib = ExistencePressLibrary()
    interaction = EncounterInteraction(seed=42)
    # Prime with high-magnitude presses (birth_event has magnitude=1.4) so
    # the receivability window mu1 settles near 1.5.
    for _ in range(10):
        interaction.process(lib.birth_event())

    press = _press_like(lib.birth_event(), magnitude=1.5,
                        label="boundary_test")
    event = interaction.process(press)

    assert event.encountered is True, (
        f"magnitude=1.5 is the boundary and must be allowed (strictly > 1.5 blocks). "
        f"Got encountered={event.encountered}, reason={event.no_encounter_reason!r}"
    )


def test_press_magnitude_below_ceiling_unaffected():
    """A normal-magnitude press behaves as before — baseline regression check.
    Light priming keeps sigma1 wide enough that magnitude=0.5 is receivable."""
    lib = ExistencePressLibrary()
    interaction = EncounterInteraction(seed=42)
    # Light priming only: 15 iterations narrow sigma1 to its floor (0.1) and
    # close the magnitude window. 5 iterations is enough to shape A_act and
    # B_base while leaving the magnitude channel responsive at 0.5.
    for _ in range(5):
        interaction.process(lib.prenatal_steady())

    press = _press_like(lib.early_postnatal_skin(), magnitude=0.5,
                        label="baseline_test")
    event = interaction.process(press)

    assert event.encountered is True, (
        f"magnitude=0.5 should encounter under light prenatal priming. "
        f"reason={event.no_encounter_reason!r}"
    )
    assert event.delta_M5 is not None
