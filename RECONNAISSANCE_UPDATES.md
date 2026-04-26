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

## Phase 2.1 landed: Output records + trace linking

**Closing commit:** [filled below after commit]

**What changed:** OutputRecord (Level 1 of the outputs ladder) and
TraceLinkGraph (Level 3) are now operational. The system records
structured responses to encounters and links traces by temporal
proximity (50% of weight, primary), geometric distance in M^5 (25%),
eigenmode alignment (15%), and significance strength (10%).
Engram-literature-grounded; not similarity-only clustering.
Coefficients are additive and sum to 1.0 — no factor is a hard
multiplicative gate.

**The full ladder, for reference:**
- Level 1 (LANDED): OutputRecord — structured encounter response
- Level 2 (existing): Trace deposit — durable residue
- Level 3 (LANDED): TraceLinkGraph — weighted edges between traces
- Level 4 (Phase 2.2): Cluster detection → CandidateSecondaryAttractors
- Level 5 (Phase 2.3): Cluster reactivation as re-press (NOT generative
  emission)
- Level 6 (Phase 2.3): Self-encounter loop (is_external=False)
- Level 7 (Phase 2.4): Confirmed secondary R* via SA6 coupling.
  Construction 7 operationalized. Clause E* verifiable.

**Architectural decision recorded:** Cluster→R* is candidate-then-
confirmation. β for formation candidate, α for confirmation. Clusters
produce CandidateSecondaryAttractors at Level 4; promotion to
recognized secondary R* happens only at Level 7 when self-encounter
/ SA6 coupling validates them. This prevents declaring every cluster
a self.

**Internal output discipline:** Level 5 must be cluster-replay-as-press,
not generative emission. The system reactivates a cluster and
reintroduces it as an internally-sourced press into encounter(). This
is what distinguishes EFT from a chatbot pretending to think.

### Structural constraint discovered: _finalize is the only correct insertion point for post-encounter wiring

During Phase 2.1 implementation, the initial wiring placement (after
event.delta_M5/Sigma/C derivation) silently failed because
event.traces_count_after and event.R_star_after_capture are populated
inside EncounterInteraction._finalize, not at the call site of
process(). Reading these fields before _finalize returns stale values
with no error raised.

The test test_links_form_between_temporally_close_traces caught this
because it asserts edges form. Without that test, the bug would have
shipped: every encounter would silently produce zero edges and
Phase 2.2 cluster detection would have run against an empty graph.

STRUCTURAL CONSTRAINT for Phase 2.2 and beyond: any code that reads
R_star_after_capture, traces_count_after, or other _finalize-populated
EncounterEvent fields MUST be wired inside _finalize after the field
captures, not at the call site of process(). The dataclass-backfill
pattern in _finalize is the only correct insertion point.

Cluster detection (Phase 2.2), self-encounter (Phase 2.3), and SA6
monitor (Phase 2.4) will all need to follow this constraint.

---

## Phase 2.2 landed: Cluster detection over trace links

**Closing commit:** [filled below after commit]

**What changed:** detect_candidate_secondary_attractors operates
over the TraceLinkGraph from Phase 2.1 to identify connected,
coherent, distinct clusters of traces. Produces
CandidateSecondaryAttractor objects. CANDIDATES ARE NOT SECONDARY
R*s. Promotion to recognized R* happens at Level 7 (Phase 2.4)
after self-encounter and SA6 coupling validate them.

**Soft distinction (architectural decision A):** A cluster
qualifies as a candidate if it is meaningfully distinct from
primary R* along EITHER spatial OR orientational axes:
- Spatial: centroid distance >= 0.75 * primary_basin_radius
- Orientation: cosine similarity to primary orientation < 0.70
Either route alone is sufficient. The dual criterion respects
that endogenous alterity may begin INSIDE the primary basin with
sub-structures forming first and differentiating later.

**Provisional thresholds (decision B):**
- MIN_CLUSTER_SIZE = 3
- MIN_MEAN_EDGE_WEIGHT = 0.30
- BASIN_RADIUS_FACTOR = 0.75
- ORIENTATION_DISTINCTION_THRESHOLD = 0.70

These are calibrated against architectural commitment (secondary
R*s are significant, not common). Revisitable if synthetic-DEAP
surfaces tuning evidence.

**Calibration finding (synthetic-DEAP at seed=42, 30 trials):**
1 candidate produced — size=20, route=both, dist=0.877, cos=-0.002,
mean_w=0.709. Below the overproduction threshold of 3 candidates;
no diagnostic flag triggered. The single candidate has high
internal coherence (mean edge weight 0.709 vs 0.30 minimum) and is
distinct from primary R* on both routes (spatial 0.877 against the
0.75·basin floor; orientation cosine ≈ 0 — nearly orthogonal).
Calibration looks tight against synthetic data; revisit after
real-DEAP run.

**What's now operational on the staged ladder:**
- Level 1 (Phase 2.1): OutputRecord
- Level 2 (existing): Trace deposit
- Level 3 (Phase 2.1): TraceLinkGraph
- Level 4 (THIS PHASE): CandidateSecondaryAttractor detection
- Level 5 (Phase 2.3): Cluster reactivation as re-press
- Level 6 (Phase 2.3): Self-encounter loop
- Level 7 (Phase 2.4): Confirmed secondary R* via SA6 coupling

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

### Cross-encounter cluster stabilization (recorded April 27)

As Phase 2.2 cluster detection lands, an architectural question
arises: does H(t)/Z(t) extend cleanly to govern cluster-stabilization
dynamics across encounters, or is a separate slower mechanism needed
on top? H(t) currently governs encounter gating and (after the §17.4
fix) some cross-encounter adaptive dynamics in α/β/μ₁. The open
question is whether trace-link cluster stabilization requires a
slower cross-encounter gate beyond H(t)/Z(t), or whether the existing
machinery should be extended to fill that role.

Reference point: 2025 Nature work on astrocytic ensembles describes
a slow integrator distinct from neuronal engrams that gates which
traces stabilize across emotional repetitions. This is a structural
parallel; whether EFT needs an analogous distinct mechanism is the
open question.

Where the work happens: a future module potentially named
eft_consolidation_gate.py if a distinct mechanism is needed;
otherwise extension of existing H(t)/Z(t) update logic in
eft_encounter_env.py.

Status: Not blocking. Surface again if Phase 2.2 cluster detection
produces noisy or over-eager candidate attractors against synthetic-
DEAP data — that would be evidence the existing machinery is
insufficient and a slower gate needs to be built.

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
