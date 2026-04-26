# RECONNAISSANCE UPDATES

Closure record for the original reconnaissance audit.

Date: 2026-04-26
Reference: `RECONNAISSANCE.pdf` (committed in `382a964`)
Branch: `claude/eft-reconnaissance-audit-gr5Sz`

This document records architectural-correction violations from the original
reconnaissance audit that have been closed by subsequent commits. The
original audit verdicts are superseded for the items below; the rest of
the audit remains current.

---

### §17.3 — R₁ overwhelm ceiling

- **Original verdict:** VIOLATED. Press magnitude > 1.5 was permitted to
  reach the encounter machinery; `ExistencePress.__post_init__` clamped
  only at the lower bound (≥ 0). Library presses `fire_extreme(2.0)` and
  `pain_extreme(1.8)` could trigger encounters whenever the gate's other
  channels supplied enough commensurability.
- **Closing commit:** `073b696` — *Fix §17.3: enforce R1 overwhelm ceiling
  at magnitude > 1.5*.
- **What changed:** The ceiling is now enforced as a visible no-encounter
  event with reason `magnitude_overwhelm` rather than as a silent clamp at
  construction time. The press still exists, is still recorded in
  `press_history` and `no_encounter_history` with full receivability data,
  but the encounter is blocked. The check fires at both layers — env and
  interaction — because the interaction layer makes its own gate decision
  before consulting `env.receive`. The boundary is strictly `>`: magnitude
  exactly equal to 1.5 is allowed.
- **Verification:** `tests/test_r1_ceiling.py` (3 tests pass: above, at,
  below ceiling). Regression: existing phase smoke test still PASS.
- **New verdict:** RESPECTED.

---

### §17.1 / §17.7 — Eigenmode pre-assignment / honesty about scaffolding

- **Original verdict:** VIOLATED on §17.7 (and PARTIAL on §17.1). The
  `EigenmodeProjector` carried a 25-entry `keyword_affinity` lookup table
  and a `_keyword_weights` method that silently fired whenever a caller
  forgot `explicit_weights`. In current flow the path was unreachable
  (the interaction layer always supplies weights), but the scaffolding
  existed and lied by existing — the canonical §17.7 failure mode.
- **Closing commit:** `3308e7f` — *Finish §17.1 / §17.7: remove
  EigenmodeProjector keyword fallback*.
- **What changed:** The `keyword_affinity` dict and `_keyword_weights`
  method are deleted. Both `EigenmodeProjector.project` and
  `EFTSystem.encounter` now raise `ValueError` when called without
  eigenmode weights, with error messages that anchor the historical
  context ("April 10 correction") and cite §17.1. Eigenmode projection is
  output-side only: real signals via DEAP physiology, or
  `press.direction × sigma_profile` from the interaction layer.
- **Verification:** `tests/test_eigenmode_projector_no_keyword.py`
  (3 tests pass: explicit weights work, missing weights raise, scaffolding
  attributes do not exist). Regression: R1 ceiling tests and phase smoke
  test still PASS.
- **New verdict:** RESPECTED.

---

### §17.4 — H2 survival floor

- **Original verdict:** VACUOUSLY HELD (effectively VIOLATED). The
  documented floors α ≥ 0.05, β ≥ 0.01 had no enforcement clamp, but the
  rule held vacuously because `update_H2` was monotone non-decreasing —
  α and β could only grow. H2 was a permanent counter, not a living
  condition-field state.
- **Closing commit:** `ff1ed41` — *Fix §17.4: H2 adaptive update with
  survival-floor clamps*.
- **What changed:** `update_H2` now mirrors `update_H1`'s "move toward
  new observation" pattern — the past must shape the present, not bury it.
  Normalized valence `u = (eps2+1)/2` defines targets `target_alpha = u`,
  `target_beta = 1−u`; α and β move both up and down toward those targets,
  scaled by `eta · novelty`. Hard clamps `max(0.05, …)` and `max(0.01, …)`
  hold the survival floors. The approach orientation can never be
  extinguished; neither can avoidance.
- **Verification:** `tests/test_h2_adaptive_floor.py` (4 tests pass: rises
  under positive valence with β falling; falls back when valence reverses;
  α clamps at 0.05 under sustained negative drive; β clamps at 0.01 under
  sustained positive drive). Regression: R1 ceiling, projector, and phase
  smoke test all still PASS.
- **New verdict:** RESPECTED.

---

### Smoke test behavioral note

The phase smoke test in `eft_encounter_interaction.py --test` continues to
pass its `event.encountered` and `dominant_eigenmode` assertions, but the
underlying H2 numbers in `learning_report()['condition_field']` look
visibly different now. Pre-fix, sustained positive valence priming
(`prenatal_steady` valence=0.5, `early_postnatal_skin` valence=0.7) pushed
both α and β upward together. Post-fix, the same priming pulls α toward
~0.6–0.75 and β downward toward ~0.25–0.4. Anyone reading the report
should expect an asymmetric approach/avoidance profile where there used
to be a symmetric one. This is the new correct behavior.

---

## Open questions surfaced during cleanup

These are not holdover items from the original audit. They emerged from
the cleanup work itself and are recorded here so future sessions can
pick them up cleanly.

### DEAP synthetic stress-event V_E asymmetry

**Surfaced by:** D-1 admissibility diagnostic, commit 07ee251.

**Finding:** Synthetic-DEAP stress event (trial P01_T04 at seed=42)
produced R* V_E motion of 0.162 while the corresponding press supplied
only 0.081 V_E direction — system V_E displacement exceeded what the
external press justified. Clause C correctly flagged this as inadmissible.

**Open question:** Determine whether this is (a) a translator calibration
issue in how DEAP physiological signals map to press.direction[V_E],
(b) an artifact of the synthetic stress-generator producing physiology
that doesn't naturally route exit-orientation into the press shape, or
(c) a deeper stress-to-exit mapping question about how the architecture
expects stress events to engage V_E.

**Where the work happens:** eft_deap_translator.py for (a) and (b);
potentially the corpus / handoff for (c).

**Status:** Not blocking. Real-DEAP-data validation (Priority 6 in the
original audit) is the natural place to revisit this — real stress
signal may resolve the question by showing whether the asymmetry
persists or is purely synthetic-generator artifact.

---

## What remains open

In priority order, items from the reconnaissance NOT yet closed:

- **Priority 1** — D-1 Revision 2 admissibility test suite. No code can
  yet verify Clauses (A)/(B)/(C)/(D)/(E*) hold for a given encounter
  event. Work would happen in a new `tests/test_d1_admissibility.py`
  plus instrumentation in `eft_encounter_interaction.py`.
- **Priority 2** — Multi-R* with SA6 sequence monitor. Depends on
  Construction 7 (endogenous alterity) being implemented; only a single
  primary R* exists today. Work spans `eft_system.py` (R*_i basins) and
  the interaction layer (sequence-level coupling check).
- **Priority 6** — DEAP integration validation suite. The end-to-end
  path (DEAP → ExistencePress → EncounterEvent) is wired in
  `eft_deap_translator.py` and its `run_demonstration`, but has no
  pytest-level coverage of `delta_M5`/`delta_Sigma`/`delta_C` population.
- **Priority 7** — `ConsolidationState` (Z) is read by `_unsettledness`
  but never written. Either implement consolidation dynamics in
  `eft_encounter_env.py` or document Z as a structural placeholder.
- **Priority 8** — `# BUG FIX` comment in `eft_system.py` lacks
  rationale. Either expand the comment to record what was fixed and why
  `input_activation` is now scaled by `w_explicit` norm, or replace it.
- **Lower-priority / longer-horizon** — Σ(S, t) promotion from
  observational monitor to constitutive PDE term; the witnessing
  architecture as a whole; the Exit Clause threshold function;
  four-belief-moments + coupling structure as code object inside
  `EncounterEvent`; U(p) parameter-uncertainty inventory per Doc 1 §7.3.
