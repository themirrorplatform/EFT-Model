"""
EFT ENCOUNTER INTERACTION
The Mirror Platform LLC - Ilya Belous - April 2026

The interaction layer inside the encounter zone.
Owns the EncounterEvent. Both subsystems update from that event.
"""
import sys
sys.path.insert(0, '/home/claude')
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from eft_verified_core import (DIM, V_T, V_A, V_B, V_R, V_E, GHOST_FLOOR, DEPOSIT_THRESHOLD)
from eft_encounter_env import (ExistencePress, ConditionFieldState, ConditionalEncounterEnvironment,
    ExistencePressLibrary, ActiveConditionField, ReceivabilityOperators, CommensurabilityGate)
from eft_outputs import OutputHistory, output_record_from_event
from eft_trace_links import TraceLinkGraph, link_new_trace

@dataclass
class EncounterEvent:
    t: int
    press: ExistencePress
    Lambda_act: Dict
    r_scores: Dict
    C_before: float
    sigma_profile: np.ndarray
    enc_args: Dict
    Sigma_before: int = 0
    self_result: Optional[Dict] = None
    Sigma_after: int = 0
    C_after: float = 0.0
    delta_M5: Optional[Dict] = None
    delta_Sigma: Optional[Dict] = None
    delta_C: Optional[Dict] = None
    encountered: bool = False
    no_encounter_reason: str = ""
    R_star_before: Optional[np.ndarray] = None
    R_star_after_capture: Optional[np.ndarray] = None
    traces_count_before: int = 0
    traces_count_after: int = 0

def _valence_to_emotion(valence):
    if valence > 0.5: return "joy"
    if valence > 0.2: return "awe"
    if valence > -0.1: return "neutral"
    if valence > -0.4: return "fear"
    if valence > -0.7: return "grief"
    return "anger"

def press_to_encounter_args(press, r_scores, lam):
    sigma = r_scores["sigma_profile"]
    weights = [float(press.direction[i] * sigma[i]) for i in range(DIM)]
    if sum(abs(w) for w in weights) < 0.05:
        weights = press.direction.tolist()
    emotion = _valence_to_emotion(press.valence)
    mu1 = lam["mu1_act"]; mag_dev = press.magnitude - mu1
    body_pert = [float(abs(mag_dev)*0.3), float(max(0.0,-mag_dev)*0.2), float(r_scores["r5"]*0.1)]
    return {"content": press.label, "eigenmode_weights": weights, "position": None,
            "body_perturbation": body_pert, "is_external": True,
            "current_emotion": emotion, "significance_override": None}

class EncounterInteraction:
    def __init__(self, seed=42):
        from eft_system import EFTSystem
        self.env = ConditionalEncounterEnvironment(seed=seed)
        self.self_system = EFTSystem(seed=seed)
        self.t = 0
        self.events: List[EncounterEvent] = []
        self.output_history = OutputHistory()
        self.trace_link_graph = TraceLinkGraph()
        self.trace_times: List[int] = []  # parallel to self.self_system.traces
        print("EncounterInteraction initialized.")

    def _finalize(self, event, R_star_before, traces_count_before):
        event.R_star_before = R_star_before
        event.traces_count_before = traces_count_before
        event.R_star_after_capture = self.self_system.R_star.copy()
        event.traces_count_after = len(self.self_system.traces)

        # Phase 2.1 limitation: EFTSystem prunes traces when count > 200.
        # When pruning happens, our parallel trace_times list and the link
        # graph become out of sync. For Phase 2.1 we accept this and reset
        # the graph on detection. Phase 2.2 cluster detection will need to
        # handle pruning more carefully (e.g., by tracking trace identity
        # rather than position-in-list).
        if len(self.self_system.traces) < len(self.trace_times):
            self.trace_link_graph = TraceLinkGraph()
            self.trace_times = [self.t for _ in self.self_system.traces]

        # Level 1: record output
        output = output_record_from_event(event)
        if output is not None:
            self.output_history.append(output)

        # Level 3: link new trace into graph if a trace was deposited
        if event.traces_count_after > event.traces_count_before:
            new_idx = len(self.self_system.traces) - 1
            self.trace_times.append(self.t)
            link_new_trace(
                self.trace_link_graph,
                self.self_system.traces, self.trace_times,
                new_idx
            )

        self.events.append(event)
        return event

    def process(self, press):
        self.t += 1
        R_star_before = self.self_system.R_star.copy()
        traces_count_before = len(self.self_system.traces)
        if press.magnitude > 1.5:
            event = EncounterEvent(t=self.t, press=press, Lambda_act={}, r_scores={},
                                   C_before=0.0, sigma_profile=np.zeros(DIM), enc_args={},
                                   Sigma_before=len(self.env.Sigma))
            event.encountered = False
            event.no_encounter_reason = (f"magnitude_overwhelm: press.magnitude={press.magnitude:.3f} "
                                         f"above R1 ceiling 1.5")
            self.env.receive(press)
            return self._finalize(event, R_star_before, traces_count_before)
        lam = ActiveConditionField().compute(self.env.H, self.env.h, self.env.B_fatigue, self.env.q)
        r_scores = ReceivabilityOperators().compute_all(press, lam)
        gate = CommensurabilityGate()
        C_result_before = gate.compute(r_scores, lam["A_act"])
        C_before = C_result_before["C"]
        Sigma_before = len(self.env.Sigma)
        enc_args = press_to_encounter_args(press, r_scores, lam)
        event = EncounterEvent(t=self.t, press=press, Lambda_act=lam, r_scores=r_scores,
                               C_before=C_before, sigma_profile=r_scores["sigma_profile"].copy(),
                               enc_args=enc_args, Sigma_before=Sigma_before)
        if not C_result_before["encounter"]:
            event.encountered = False
            event.no_encounter_reason = (f"C={round(C_before,3)} below theta_e={C_result_before['theta_e']}"
                if not C_result_before["above_threshold"] else "support condition not met")
            self.env.receive(press)
            return self._finalize(event, R_star_before, traces_count_before)
        event.encountered = True
        self.env.receive(press)
        event.self_result = self.self_system.encounter(**enc_args)
        event.Sigma_after = len(self.env.Sigma)
        lam_after = ActiveConditionField().compute(self.env.H, self.env.h, self.env.B_fatigue, self.env.q)
        r_scores_after = ReceivabilityOperators().compute_all(press, lam_after)
        C_result_after = gate.compute(r_scores_after, lam_after["A_act"])
        event.C_after = C_result_after["C"]
        event.delta_M5 = self._derive_delta_M5(event)
        event.delta_Sigma = self._derive_delta_Sigma(event)
        event.delta_C = self._derive_delta_C(event)
        return self._finalize(event, R_star_before, traces_count_before)

    def _derive_delta_M5(self, event):
        sr = event.self_result
        return {"mode": sr["mode"], "significance": sr["significance"],
                "primary_eigenmode": sr["primary_eigenmode"],
                "eigenmode_impact": sr["eigenmode_impact"],
                "det_G_ratio": sr["det_G_ratio"],
                "R_star_after": sr["R_star"],
                "R_star_dominant": sr["R_star_eigenmode_position"]}

    def _derive_delta_Sigma(self, event):
        new_entries = event.Sigma_after - event.Sigma_before
        actualized = (event.sigma_profile * event.press.direction)
        return {"new_entries": new_entries, "Sigma_before": event.Sigma_before,
                "Sigma_after": event.Sigma_after,
                "actualized_direction": actualized.round(4).tolist(),
                "actualized_magnitude": round(float(event.press.magnitude * event.r_scores["r1"]),4),
                "actualized_valence": round(float(event.press.valence * event.r_scores["r2"]),4)}

    def _derive_delta_C(self, event):
        names = ["temporal","agency","boundary","reality","exit"]
        eigvals = np.linalg.eigvalsh(self.env.H.A)
        return {"C_before": round(event.C_before,4), "C_after": round(event.C_after,4),
                "C_shift": round(event.C_after-event.C_before,4),
                "C_shift_direction": "opened" if event.C_after > event.C_before else "closed",
                "dominant_eigenmode_after": names[int(np.argmax(eigvals))],
                "mu1_after": round(self.env.H.mu1,4), "alpha_after": round(self.env.H.alpha,4),
                "beta_after": round(self.env.H.beta,4),
                "unsettledness": round(self.env._unsettledness(),4)}

    def learning_report(self):
        H = self.env.H; names = ["temporal","agency","boundary","reality","exit"]
        eigvals = np.linalg.eigvalsh(H.A)
        encountered = [e for e in self.events if e.encountered]
        modes = {}
        for e in encountered:
            if e.delta_M5:
                m = e.delta_M5["mode"]; modes[m] = modes.get(m,0)+1
        return {"total_presses": len(self.events), "encounters": len(encountered),
                "encounter_rate": round(len(encountered)/max(1,len(self.events)),3),
                "mode_distribution": modes, "actualization_record": len(self.env.Sigma),
                "condition_field": {"magnitude_center": round(H.mu1,4),
                    "approach_alpha": round(H.alpha,4), "avoidance_beta": round(H.beta,4),
                    "dominant_eigenmode": names[int(np.argmax(eigvals))],
                    "weakest_eigenmode": names[int(np.argmin(eigvals))],
                    "anisotropy": {names[i]: round(float(eigvals[i]),4) for i in range(5)}},
                "geometric": {"R_star": {names[i]: round(float(self.self_system.R_star[i]),4) for i in range(5)},
                    "R_star_dominant": names[int(np.argmax(np.abs(self.self_system.R_star)))],
                    "total_traces": len(self.self_system.traces)},
                "trajectory": self.self_system.sigma.summary()}

    def self_observe(self):
        env_result = self.env.self_observe()
        so = env_result.get("self_observation",{})
        return {"occurred": env_result["encounter_result"]["occurred"],
                "C": env_result["commensurability"]["C"],
                "sees_clearly": so.get("dominant_eigenmode","unknown"),
                "dominant_lambda": so.get("dominant_eigenvalue",0.0),
                "blind_to": so.get("blind_direction","unknown"),
                "blind_lambda": so.get("min_eigenvalue",0.0),
                "note": "Self-observation updated H(t). Next observation will differ."}

def run_phase_tests():
    lib = ExistencePressLibrary()
    print("\n--- Phase 1: Translation ---")
    env = ConditionalEncounterEnvironment(seed=42)
    for _ in range(10): env.receive(lib.prenatal_steady())
    press = lib.early_postnatal_skin()
    lam = ActiveConditionField().compute(env.H, env.h, env.B_fatigue, env.q)
    r = ReceivabilityOperators().compute_all(press, lam)
    args = press_to_encounter_args(press, r, lam)
    assert sum(abs(w) for w in args["eigenmode_weights"]) > 0.05
    assert args["current_emotion"] in ["joy","awe","neutral","fear","grief","anger"]
    print("Phase 1: PASS")
    print("\n--- Phase 2: Event Ownership ---")
    interaction = EncounterInteraction(seed=42)
    for _ in range(15): interaction.process(lib.prenatal_steady())
    event = interaction.process(lib.early_postnatal_skin())
    assert event.encountered, "Skin contact should encounter after prenatal priming"
    assert event.delta_M5 is not None
    assert event.delta_Sigma is not None
    assert event.delta_C is not None
    assert event.delta_C["C_before"] != event.delta_C["C_after"]
    assert event.delta_Sigma["Sigma_after"] > event.delta_Sigma["Sigma_before"]
    print("Phase 2: PASS")
    print("\n--- Phase 3: Learning ---")
    report = interaction.learning_report()
    assert report["encounters"] > 0
    assert report["condition_field"]["dominant_eigenmode"] != ""
    assert report["actualization_record"] > 0
    so = interaction.self_observe()
    assert so["sees_clearly"] != ""
    assert so["blind_to"] != ""
    print("Phase 3: PASS")
    print("\nAll phase tests passed.")
    return interaction

if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        run_phase_tests()
    else:
        lib = ExistencePressLibrary(); i = EncounterInteraction(seed=42)
        for _ in range(20): i.process(lib.prenatal_steady())
        i.process(lib.birth_event())
        for _ in range(5): i.process(lib.early_postnatal_skin())
        r = i.learning_report()
        print(f"Encounters: {r['encounters']}/{r['total_presses']}")
        print(f"Dominant: {r['condition_field']['dominant_eigenmode']}")
        print(r['trajectory'])
