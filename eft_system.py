"""
EFT SYSTEM -- COMPLETE RUNNING IMPLEMENTATION
The Mirror Platform LLC - Ilya Belous - April 2026
"""
import sys
sys.path.insert(0, '/home/claude')
import numpy as np
from scipy.linalg import expm, logm
from scipy.optimize import minimize
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import json
import warnings
warnings.filterwarnings('ignore')

from eft_verified_core import (
    DIM, GHOST_FLOOR, DEPOSIT_THRESHOLD, LAMBDA_DECAY, KAPPA, BETA,
    SIGMA_MIN, SIGMA_MAX, S_THRESH, T_SIG, B_DIM, H_DIM,
    V_T, V_A, V_B, V_R, V_E, D4_INDICES, EXIT_INDEX, ALIGNMENT_REGEN,
    EIGENMODE_DIRS, BodySubstrate, Trace,
    build_S_traces, build_G_total, Psi, find_R_star, compute_Hessian_at_R_star,
    sigma_from_salience, SignificanceFunctional, ConflictOperator, EvolutionRateTensor,
    TauIrrTracker
)

PROVISIONAL_W_DOMAIN = {
    'fear':     np.array([0.1, 0.2, 0.8, 0.3, 0.6]),
    'grief':    np.array([0.7, 0.2, 0.6, 0.2, 0.3]),
    'anger':    np.array([0.3, 0.8, 0.4, 0.3, 0.2]),
    'shame':    np.array([0.2, 0.2, 0.3, 0.8, 0.1]),
    'joy':      np.array([0.5, 0.5, 0.5, 0.5, 0.5]),
    'awe':      np.array([0.3, 0.3, 0.3, 0.7, 0.8]),
    'disgust':  np.array([0.1, 0.3, 0.9, 0.4, 0.1]),
    'surprise': np.array([0.4, 0.4, 0.4, 0.6, 0.6]),
    'neutral':  np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
}

EIGENMODE_NAMES = {V_T: 'temporal', V_A: 'agency', V_B: 'boundary', V_R: 'reality', V_E: 'exit'}

VELD_VOCAB = {
    'hasveld': 'permanent-past-trace', 'dhruvat': 'attractor-present',
    'chuveld': 'minimum-encounter', 'vasveld': 'trace', 'karveld': 'active-trace',
    'gnoveld': 'generative-boundary', 'qiveld': 'affective-metric',
    'dhruveld': 'self-attractor', 'jingveld': 'stable-neighborhood',
    'mode_gen': 'generative-encounter', 'mode_regen': 'regenerative-encounter',
    'mode_neut': 'neutral-encounter', 'mode_deg': 'degenerate-encounter',
}


class EigenmodeProjector:
    def __init__(self):
        self.learned_weights: Dict[str, np.ndarray] = {}
        self.encounter_count = 0
        self.keyword_affinity = {
            'past': np.array([0.9,0.1,0.1,0.1,0.1]), 'memory': np.array([0.8,0.1,0.1,0.2,0.1]),
            'history': np.array([0.9,0.1,0.1,0.1,0.1]), 'before': np.array([0.8,0.1,0.1,0.1,0.1]),
            'was': np.array([0.7,0.1,0.1,0.1,0.1]), 'identity': np.array([0.6,0.2,0.3,0.2,0.1]),
            'act': np.array([0.1,0.9,0.1,0.1,0.1]), 'do': np.array([0.1,0.8,0.1,0.1,0.1]),
            'choose': np.array([0.1,0.8,0.2,0.2,0.2]), 'control': np.array([0.1,0.7,0.3,0.1,0.1]),
            'can': np.array([0.1,0.8,0.1,0.1,0.1]), 'protect': np.array([0.1,0.2,0.9,0.1,0.1]),
            'edge': np.array([0.1,0.1,0.8,0.2,0.2]), 'limit': np.array([0.1,0.2,0.7,0.3,0.1]),
            'inside': np.array([0.2,0.1,0.8,0.1,0.1]), 'outside': np.array([0.1,0.1,0.6,0.2,0.4]),
            'other': np.array([0.1,0.1,0.5,0.2,0.5]), 'true': np.array([0.1,0.1,0.1,0.9,0.1]),
            'real': np.array([0.1,0.1,0.1,0.9,0.1]), 'test': np.array([0.1,0.2,0.1,0.8,0.1]),
            'believe': np.array([0.2,0.1,0.2,0.7,0.1]), 'know': np.array([0.2,0.1,0.1,0.8,0.1]),
            'leave': np.array([0.1,0.1,0.2,0.1,0.9]), 'beyond': np.array([0.1,0.1,0.2,0.2,0.8]),
            'unknown': np.array([0.1,0.1,0.1,0.2,0.9]),
        }

    def project(self, content, explicit_weights=None, explicit_position=None):
        weights = np.array(explicit_weights, dtype=float) if explicit_weights is not None else self._keyword_weights(content)
        w_norm = np.linalg.norm(weights)
        if w_norm < 1e-12:
            weights = np.ones(DIM) / np.sqrt(DIM); w_norm = 1.0
        n_p = weights / w_norm
        x_p = np.array(explicit_position, dtype=float) if explicit_position is not None else n_p * min(w_norm * 0.3, 1.5)
        self.encounter_count += 1
        return x_p, n_p

    def _keyword_weights(self, content):
        words = content.lower().split()
        weights = np.zeros(DIM)
        for word in words:
            if word in self.keyword_affinity:
                weights += self.keyword_affinity[word]
            else:
                for key in self.keyword_affinity:
                    if key in word or word in key:
                        weights += self.keyword_affinity[key] * 0.5; break
            if word in self.learned_weights:
                weights += self.learned_weights[word]
        if np.max(weights) < 1e-10:
            weights = np.ones(DIM) * 0.2
        return weights

    def learn_from_encounter(self, content, actual_n_p, significance):
        if significance < GHOST_FLOOR: return
        words = content.lower().split()
        for word in words:
            if word not in self.learned_weights:
                self.learned_weights[word] = np.zeros(DIM)
            lr = min(significance * 0.1, 0.3)
            self.learned_weights[word] += lr * (actual_n_p - self.learned_weights[word])


@dataclass
class TrajectorySignature:
    R_star_history: List[np.ndarray] = field(default_factory=list)
    mode_counts: Dict[str, int] = field(default_factory=lambda: {'generative':0,'regenerative':0,'neutral':0,'degenerate':0})
    hessian_min_eigenvalue_history: List[float] = field(default_factory=list)
    v_E_access_count: int = 0
    total_encounters: int = 0
    gamma_estimate: float = 0.0
    time: int = 0

    def update(self, R_star, mode, hessian_min_eig, encounter_is_external, significance):
        self.R_star_history.append(R_star.copy())
        if len(self.R_star_history) > 500: self.R_star_history.pop(0)
        if mode in self.mode_counts: self.mode_counts[mode] += 1
        self.hessian_min_eigenvalue_history.append(hessian_min_eig)
        if len(self.hessian_min_eigenvalue_history) > 100: self.hessian_min_eigenvalue_history.pop(0)
        self.total_encounters += 1; self.time += 1
        if mode == 'generative' and encounter_is_external: self.v_E_access_count += 1
        if not encounter_is_external and significance > GHOST_FLOOR:
            self.gamma_estimate = min(self.gamma_estimate * 0.9 + 0.1, 1.0)
        elif encounter_is_external and significance > GHOST_FLOOR:
            self.gamma_estimate = max(self.gamma_estimate * 0.8, 0.0)

    def drift_signals(self):
        signals = {}
        if len(self.R_star_history) >= 5:
            recent = np.array(self.R_star_history[-5:])
            d = float(np.std(recent, axis=0).mean())
            signals['R_star_drift'] = d; signals['R_star_drift_warning'] = d > 0.3
        else:
            signals['R_star_drift'] = 0.0; signals['R_star_drift_warning'] = False
        if self.total_encounters > 10:
            mf = {k: v/self.total_encounters for k,v in self.mode_counts.items()}
            signals['mode_distribution'] = mf; signals['mode_collapse_warning'] = max(mf.values()) > 0.85
        else:
            signals['mode_distribution'] = self.mode_counts.copy(); signals['mode_collapse_warning'] = False
        if len(self.hessian_min_eigenvalue_history) >= 3:
            rh = self.hessian_min_eigenvalue_history[-10:]
            signals['hessian_min_eig'] = float(np.mean(rh)); signals['stability_warning'] = float(np.mean(rh)) < 0.05
        else:
            signals['hessian_min_eig'] = 1.0; signals['stability_warning'] = False
        if self.total_encounters > 0:
            vr = self.v_E_access_count / self.total_encounters
            signals['v_E_access_rate'] = vr; signals['exit_atrophy_warning'] = vr < 0.05 and self.total_encounters > 20
        else:
            signals['v_E_access_rate'] = 0.0; signals['exit_atrophy_warning'] = False
        signals['gamma_estimate'] = self.gamma_estimate; signals['E3_violation_warning'] = self.gamma_estimate > 0.8
        return signals

    def summary(self):
        ds = self.drift_signals()
        w = [k for k,v in [('R* DRIFTING',ds.get('R_star_drift_warning')),('MODE COLLAPSE',ds.get('mode_collapse_warning')),
             ('STABILITY LOSS',ds.get('stability_warning')),('EXIT ATROPHY',ds.get('exit_atrophy_warning')),
             ('E3 VIOLATION RISK',ds.get('E3_violation_warning'))] if v]
        status = 'STABLE' if not w else ' | '.join(w)
        return f"Sigma[t={self.time}]: {status} | encounters={self.total_encounters} | modes={self.mode_counts} | gamma={self.gamma_estimate:.3f}"


class EFTSystem:
    def __init__(self, seed=0, persistent_state=None):
        np.random.seed(seed)
        self.body = BodySubstrate(seed=seed)
        self.traces: List[Trace] = []
        self.trace_id_counter = 0
        self.time = 0
        self.sig_functional = SignificanceFunctional()
        self.ep_tensor = EvolutionRateTensor()
        self.tau_irr = TauIrrTracker()
        self.projector = EigenmodeProjector()
        self.sigma = TrajectorySignature()
        self.R_star = np.zeros(DIM)
        self.R_star_initialized = False
        self.learned_w_domain: Dict[int, np.ndarray] = {}
        self.default_w_domain = np.ones(DIM) * 0.2
        self.h_0 = np.zeros(H_DIM)
        self.encounter_history: List[Dict] = []
        if persistent_state: self._load_state(persistent_state)

    def encounter(self, content='', eigenmode_weights=None, position=None,
                  body_perturbation=None, is_external=True, current_emotion='neutral',
                  significance_override=None):
        self.time += 1
        b_pert = np.array(body_perturbation[:B_DIM], dtype=float) if body_perturbation else None
        h_current = self.body.step(b_pert)
        w_explicit = np.array(eigenmode_weights, dtype=float) if eigenmode_weights else None
        p_explicit = np.array(position, dtype=float) if position else None
        x_enc, n_enc = self.projector.project(content, w_explicit, p_explicit)
        G_field = self.body.G_field(); G_body = self.body.G_body()
        G_before = build_G_total(self.traces, x_enc, G_field, G_body)

        # BUG FIX: scale by input activation, not constant
        temp_trace = Trace(position=x_enc.copy(), significance=1.0,
                           kernel_width=sigma_from_salience(1.0), normal=n_enc.copy(), deposit_time=self.time)
        G_probe = build_G_total(self.traces + [temp_trace], x_enc, G_field, G_body)
        if w_explicit is not None:
            input_activation = float(np.linalg.norm(w_explicit))
        else:
            raw_w = self.projector._keyword_weights(content)
            input_activation = max(float(np.linalg.norm(raw_w)) * 0.15, 0.3)

        if significance_override is not None:
            sig_value = significance_override
            sig_data = {'significance': sig_value, 'delta_G_norm': float(np.linalg.norm(G_probe - G_before,'fro')),
                        'I_SB':1.0,'P_tau_lambda':1.0,'above_ghost_floor':sig_value>=GHOST_FLOOR,
                        'above_deposit_threshold':sig_value>=DEPOSIT_THRESHOLD}
        else:
            raw_sig = self.sig_functional.compute(G_before, G_probe, x_enc, self.traces, G_field, G_body)
            scaled = raw_sig['delta_G_norm'] * input_activation
            sig_value = scaled * raw_sig['P_tau_lambda'] * raw_sig['I_SB']
            sig_data = dict(raw_sig); sig_data['delta_G_norm'] = scaled; sig_data['significance'] = sig_value
            sig_data['above_ghost_floor'] = sig_value >= GHOST_FLOOR
            sig_data['above_deposit_threshold'] = sig_value >= DEPOSIT_THRESHOLD

        if not is_external and sig_value < GHOST_FLOOR: mode = 'degenerate'
        elif sig_value < DEPOSIT_THRESHOLD: mode = 'neutral'
        else:
            max_a = max((abs(float(np.dot(n_enc, t.normal))) for t in self.traces), default=0.0)
            mode = 'regenerative' if max_a >= ALIGNMENT_REGEN else 'generative'

        trace_deposited = False
        if sig_value >= DEPOSIT_THRESHOLD:
            self.traces.append(Trace(position=x_enc.copy(), significance=sig_value,
                                     kernel_width=sigma_from_salience(sig_value),
                                     normal=n_enc.copy(), deposit_time=self.time, current_age=0))
            self.trace_id_counter += 1; trace_deposited = True
            ew = PROVISIONAL_W_DOMAIN.get(current_emotion, self.default_w_domain)
            self.learned_w_domain[self.trace_id_counter] = ew.copy()
            self.projector.learn_from_encounter(content, n_enc, sig_value)

        G_after = build_G_total(self.traces, x_enc, G_field, G_body)
        ep_per_domain = np.zeros(DIM)
        if trace_deposited and self.traces:
            hd = self._get_hessian_diagonal()
            ew = PROVISIONAL_W_DOMAIN.get(current_emotion, self.default_w_domain)
            for i, trace in enumerate(self.traces[-min(20,len(self.traces)):]):
                tid = max(0, len(self.traces)-20+i)
                wd = self.learned_w_domain.get(tid, ew)
                ep_r = self.ep_tensor.compute(s_enc=sig_value, x_p=trace.position, p_enc=x_enc,
                    n_enc=n_enc, n_p=trace.normal, h_t=h_current, h_0=self.h_0,
                    w_domain_p=wd, rho_p=0.5, Hess_Psi_np=hd[int(np.argmax(np.abs(trace.normal)))],
                    tau_p=float(trace.current_age), lambda_p=LAMBDA_DECAY)
                ep_per_domain[int(np.argmax(np.abs(trace.normal)))] += ep_r['E_p']
                self.tau_irr.update(f"trace_{trace.deposit_time}", ep_r['E_p'])

        if len(self.traces) > 0 and (trace_deposited or not self.R_star_initialized):
            self.R_star = self._find_R_star_fast(G_field, G_body); self.R_star_initialized = True

        hessian_min_eig = 0.0; hessian_eigs = np.zeros(DIM)
        if self.R_star_initialized and len(self.traces) >= 3:
            try:
                H = compute_Hessian_at_R_star(self.R_star, self.traces, G_field, G_body, eps=0.05)
                hessian_eigs = np.sort(np.linalg.eigvalsh(H)); hessian_min_eig = float(hessian_eigs[0])
            except: pass

        self.sigma.update(self.R_star, mode, hessian_min_eig, is_external, sig_value)
        for trace in self.traces: trace.current_age += 1
        if len(self.traces) > 200:
            self.traces.sort(key=lambda t: t.effective_significance(), reverse=True)
            self.traces = self.traces[:200]

        report = self._build_report(content, mode, sig_data, sig_value, x_enc, n_enc,
                                    G_before, G_after, ep_per_domain, hessian_eigs,
                                    hessian_min_eig, trace_deposited, is_external, current_emotion)
        self.encounter_history.append(report)
        return report

    def _find_R_star_fast(self, G_field, G_body):
        def psi_full(x): return Psi(x, self.traces, G_field, G_body) + BETA/2*float(np.dot(x,x))
        def psi_grad(x):
            grad = np.zeros(DIM); eps = 5e-5
            for i in range(DIM):
                xp=x.copy(); xp[i]+=eps; xm=x.copy(); xm[i]-=eps
                grad[i] = (psi_full(xp)-psi_full(xm))/(2*eps)
            return grad
        starts = [self.R_star.copy(), np.zeros(DIM)]
        if self.traces:
            pos = np.array([t.position for t in self.traces])
            wts = np.array([t.effective_significance() for t in self.traces])
            wts /= wts.sum()+1e-12; starts.append(np.sum(wts[:,None]*pos,axis=0)*0.5)
        best, bv = None, float('inf')
        for x0 in starts:
            try:
                r = minimize(psi_full, x0, jac=psi_grad, method='L-BFGS-B', options={'maxiter':100,'ftol':1e-10})
                if r.fun < bv: bv=r.fun; best=r.x
            except: pass
        return best if best is not None else self.R_star.copy()

    def _get_hessian_diagonal(self):
        if not self.R_star_initialized or len(self.traces) < 2: return np.ones(DIM)*0.1
        G_field=self.body.G_field(); G_body=self.body.G_body()
        diag=np.zeros(DIM); eps=0.05; pc=Psi(self.R_star,self.traces,G_field,G_body)
        for i in range(DIM):
            xp=self.R_star.copy(); xp[i]+=eps; xm=self.R_star.copy(); xm[i]-=eps
            diag[i]=(Psi(xp,self.traces,G_field,G_body)-2*pc+Psi(xm,self.traces,G_field,G_body))/(eps*eps)
        return np.abs(diag)

    def _build_report(self, content, mode, sig_data, sig_value, x_enc, n_enc,
                      G_before, G_after, ep_per_domain, hessian_eigs,
                      hessian_min_eig, trace_deposited, is_external, current_emotion):
        delta_G = G_after - G_before
        eigenmode_impact = {EIGENMODE_NAMES[k]: round(float(np.eye(DIM)[k]@delta_G@np.eye(DIM)[k]),6) for k in range(DIM)}
        primary_mode_idx = int(np.argmax(np.abs(n_enc)))
        max_a = max((abs(float(np.dot(n_enc,t.normal))) for t in self.traces), default=0.0)
        det_r = float(np.linalg.det(G_after))/(float(np.linalg.det(G_before))+1e-12)
        return {
            'time': self.time, 'content': content, 'mode': mode,
            'significance': round(sig_value,6), 'above_deposit_threshold': sig_value>=DEPOSIT_THRESHOLD,
            'trace_deposited': trace_deposited,
            'position_in_M5': x_enc.round(4).tolist(), 'direction_n_p': n_enc.round(4).tolist(),
            'primary_eigenmode': EIGENMODE_NAMES[primary_mode_idx],
            'eigenmode_impact': eigenmode_impact, 'det_G_ratio': round(det_r,6),
            'evolution_rate_per_domain': {EIGENMODE_NAMES[k]: round(float(ep_per_domain[k]),6) for k in range(DIM)},
            'R_star': self.R_star.round(4).tolist(),
            'R_star_eigenmode_position': {EIGENMODE_NAMES[k]: round(float(self.R_star[k]),4) for k in range(DIM)},
            'hessian_min_eigenvalue': round(hessian_min_eig,6),
            'alignment_with_existing': round(max_a,4),
            'K_T_holdings': self._compute_K_holdings(),
            'trajectory_summary': self.sigma.summary(),
            'drift_signals': self.sigma.drift_signals(),
            'total_traces': len(self.traces), 'current_emotion': current_emotion,
            'significance_components': {
                'delta_G_norm': round(sig_data.get('delta_G_norm',0.0),6),
                'I_SB': round(sig_data.get('I_SB',1.0),6),
                'P_tau_lambda': round(sig_data.get('P_tau_lambda',1.0),6),
            },
        }

    def _compute_K_holdings(self):
        holdings = []
        active = [t for t in self.traces if t.effective_significance() > DEPOSIT_THRESHOLD]
        for i in range(len(active)):
            for j in range(i+1,len(active)):
                a = float(np.dot(active[i].normal, active[j].normal))
                if a < -0.3:
                    holdings.append({'alignment': round(a,4),
                        'tension_eigenmodes': (EIGENMODE_NAMES[int(np.argmax(np.abs(active[i].normal)))],
                                               EIGENMODE_NAMES[int(np.argmax(np.abs(active[j].normal)))])})
        return holdings

    def save_state(self):
        return {'time': self.time, 'R_star': self.R_star.tolist(), 'R_star_initialized': self.R_star_initialized,
                'traces': [{'position':t.position.tolist(),'significance':t.significance,
                             'kernel_width':t.kernel_width,'normal':t.normal.tolist(),
                             'deposit_time':t.deposit_time,'current_age':t.current_age} for t in self.traces]}

    def _load_state(self, state):
        self.time = state.get('time',0); self.R_star = np.array(state.get('R_star',np.zeros(DIM)))
        self.R_star_initialized = state.get('R_star_initialized',False); self.traces = []
        for td in state.get('traces',[]):
            self.traces.append(Trace(position=np.array(td['position']),significance=td['significance'],
                kernel_width=td['kernel_width'],normal=np.array(td['normal']),
                deposit_time=td['deposit_time'],current_age=td['current_age']))

    def report_state(self):
        lines = ["="*60,"EFT SYSTEM STATE","="*60,f"Time: {self.time}",f"Traces: {len(self.traces)}","",
                 "R* (self-attractor):"]
        for k in range(DIM): lines.append(f"  {EIGENMODE_NAMES[k]:10s}: {self.R_star[k]:+.4f}")
        lines += ["", self.sigma.summary(), "="*60]
        return "\n".join(lines)

    def format_report(self, report, verbose=True):
        lines = ["","─"*56,f"ENCOUNTER [{report['time']}]: {str(report['content'])[:60]}","─"*56,
                 f"MODE: {report['mode'].upper()}  SIG: {report['significance']:.4f}",
                 f"PRIMARY: {report['primary_eigenmode']}  TRACE: {'DEPOSITED' if report['trace_deposited'] else 'below threshold'}",
                 "","EIGENMODE IMPACT:"]
        for name,val in report['eigenmode_impact'].items():
            lines.append(f"  {name:10s}: {val:+.6f}")
        lines += ["",f"R*: {report['R_star_eigenmode_position']}",
                  f"TRAJECTORY: {report['trajectory_summary']}",""]
        return "\n".join(lines)
