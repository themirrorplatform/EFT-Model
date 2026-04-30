"""
EFT CONDITIONAL ENCOUNTER ENVIRONMENT
The Mirror Platform LLC - Ilya Belous - April 2026
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')
from eft_verified_core import (DIM, GHOST_FLOOR, DEPOSIT_THRESHOLD, V_T, V_A, V_B, V_R, V_E, EIGENMODE_DIRS)

@dataclass
class ExistencePress:
    magnitude: float
    valence: float
    direction: np.ndarray
    rhythm: np.ndarray
    persistence: float
    label: str = ""
    press_class: str = "unknown"
    def __post_init__(self):
        n = np.linalg.norm(self.direction)
        self.direction = self.direction / n if n > 1e-10 else np.array([1.,0,0,0,0])
        s = np.sum(self.rhythm)
        self.rhythm = self.rhythm / s if s > 1e-10 else np.ones(len(self.rhythm)) / len(self.rhythm)
        self.valence = float(np.clip(self.valence, -1.0, 1.0))
        self.magnitude = max(0.0, float(self.magnitude))
        self.persistence = max(0.0, float(self.persistence))

@dataclass
class ConditionFieldState:
    mu1: float = 0.5; sigma1: float = 0.4; sigma1_min: float = 0.1
    alpha: float = 0.5; beta: float = 0.5
    A: np.ndarray = field(default_factory=lambda: np.eye(DIM)/5.0)
    A_floor: float = 0.01; m_bins: int = 8
    B_base: np.ndarray = field(default_factory=lambda: np.ones(8)/8.0)
    tau_hold_base: float = 1.0
    def __post_init__(self): self._project_A(); self.B_base = self.B_base / (np.sum(self.B_base)+1e-10)
    def _project_A(self):
        self.A = (self.A + self.A.T) / 2
        eigvals, eigvecs = np.linalg.eigh(self.A)
        eigvals = np.maximum(eigvals, self.A_floor)
        self.A = eigvecs @ np.diag(eigvals) @ eigvecs.T
        tr = np.trace(self.A)
        if tr > 1e-10: self.A = self.A / tr

@dataclass
class ConsolidationState:
    mu1: float = 0.5; sigma1: float = 0.4; alpha: float = 0.5; beta: float = 0.5
    A: np.ndarray = field(default_factory=lambda: np.eye(DIM)/5.0)
    B_base: np.ndarray = field(default_factory=lambda: np.ones(8)/8.0)
    tau_hold_base: float = 1.0

class ActiveConditionField:
    delta_dep=0.3; delta_stress=0.5; delta_val=0.4; delta_overwhelm=3.0
    h_overwhelm=2.0; omega_c=4.0; delta_agg=0.3; delta_fat=0.2; delta_stress_F5=0.3

    def compute(self, H, h, B_fatigue, q):
        h_agg = float(np.linalg.norm(h))/np.sqrt(DIM)
        h_dep = max(0.0, -(h[V_A]+h[V_R])/2.0)
        h_val = float(h[V_A]-h[V_E])
        mu1_act = H.mu1 - self.delta_dep * h_dep
        sigma1_act = max(H.sigma1_min, H.sigma1 * np.exp(-self.delta_stress * h_agg))
        alpha_act = H.alpha * np.exp(self.delta_val * max(0.0, h_val))
        beta_act = H.beta * np.exp(-self.delta_val * min(0.0, h_val))
        a = 1.0 + np.log(1.0 + np.exp(h)) - np.log(2.0)
        M_h = np.diag(a)
        overwhelm_s = 1.0/(1.0+np.exp(-self.delta_overwhelm*(np.linalg.norm(h)-self.h_overwhelm)))
        A_raw = (1.0-overwhelm_s)*M_h@H.A@M_h.T + overwhelm_s*(np.trace(H.A)/DIM)*np.eye(DIM)
        tr = np.trace(A_raw)
        A_act = A_raw*(np.trace(H.A)/tr) if tr > 1e-10 else H.A.copy()
        eigvals, eigvecs = np.linalg.eigh(A_act)
        eigvals = np.maximum(eigvals, H.A_floor)
        A_act = eigvecs @ np.diag(eigvals) @ eigvecs.T
        m = len(H.B_base); freqs = np.arange(m)
        L_dep = np.exp(-h_dep*freqs/self.omega_c)
        G_agg = 1.0+self.delta_agg*h_agg*(freqs/self.omega_c)**2/(1+(freqs/self.omega_c)**2)
        N_fat = np.maximum(1.0-B_fatigue[:m]*self.delta_fat, 0.0)
        B_raw = H.B_base*L_dep*G_agg*N_fat
        B_sum = np.sum(B_raw)
        B_act = B_raw/B_sum if B_sum > 1e-10 else H.B_base.copy()
        p_stress = self.delta_stress_F5 * np.tanh(h_agg/1.0)
        tau_hold_act = max(H.tau_hold_base * q * (1.0-p_stress), 0.01)
        return {'mu1_act':mu1_act,'sigma1_act':sigma1_act,'alpha_act':alpha_act,'beta_act':beta_act,
                'A_act':A_act,'B_act':B_act,'tau_hold_act':tau_hold_act}

class ReceivabilityOperators:
    def R1_magnitude(self, eps1, mu1_act, sigma1_act):
        return float(np.exp(-(eps1-mu1_act)**2/(2*sigma1_act**2)))
    def R2_valence(self, eps2, alpha_act, beta_act):
        u = float(np.clip((eps2+1.0)/2.0, 1e-6, 1.0-1e-6))
        a = max(alpha_act, 0.1); b = max(beta_act, 0.1)
        from scipy.special import betaln
        log_val = (a-1)*np.log(u)+(b-1)*np.log(1-u)-betaln(a,b)
        mode_u = float(np.clip((a-1)/(a+b-2), 1e-6, 1.0-1e-6)) if (a+b)>2 else 0.5
        log_mode = (a-1)*np.log(mode_u)+(b-1)*np.log(1-mode_u)-betaln(a,b)
        return float(np.clip(np.exp(log_val-log_mode), 0.0, 1.0))
    def R3_direction(self, eps3, A_act):
        eps3_unit = eps3/(np.linalg.norm(eps3)+1e-12)
        num = float(eps3_unit@A_act@eps3_unit)
        lmax = float(np.max(np.linalg.eigvalsh(A_act)))
        return float(np.clip(num/lmax, 0.0, 1.0)) if lmax > 1e-10 else 0.0
    def R4_rhythm(self, eps4, B_act):
        m = min(len(eps4), len(B_act)); power = eps4[:m]**2
        total = float(np.sum(power))
        return float(np.clip(np.sum(power*B_act[:m])/total, 0.0, 1.0)) if total > 1e-12 else 0.0
    def R5_persistence(self, eps5, tau_hold_act, eps1, mu1_act, sigma1_act):
        base = 1.0-float(np.exp(-eps5/(tau_hold_act+1e-10)))
        r1 = self.R1_magnitude(eps1, mu1_act, sigma1_act)
        return float(np.clip(base*r1, 0.0, 1.0))
    def compute_all(self, press, lam):
        r1=self.R1_magnitude(press.magnitude,lam['mu1_act'],lam['sigma1_act'])
        r2=self.R2_valence(press.valence,lam['alpha_act'],lam['beta_act'])
        r3=self.R3_direction(press.direction,lam['A_act'])
        r4=self.R4_rhythm(press.rhythm,lam['B_act'])
        r5=self.R5_persistence(press.persistence,lam['tau_hold_act'],press.magnitude,lam['mu1_act'],lam['sigma1_act'])
        return {'r1':r1,'r2':r2,'r3':r3,'r4':r4,'r5':r5,'sigma_profile':np.array([r1,r2,r3,r4,r5])}

class CommensurabilityGate:
    def __init__(self, theta_e=0.35, k=2, theta_min=0.1, weights=None):
        self.theta_e=theta_e; self.k=k; self.theta_min=theta_min
        self.W = weights if weights is not None else np.ones(5)/5.0
    def compute(self, r, A_act):
        eigs = np.linalg.eigvalsh(A_act)
        W_dir = eigs/(np.sum(eigs)+1e-10)
        W = 0.5*self.W + 0.5*W_dir; W = W/(np.sum(W)+1e-10)
        sigma = r['sigma_profile']; C = float(np.sum(W*sigma))
        above_min = int(np.sum(sigma > self.theta_min)); support_met = above_min >= self.k
        encounter = (C > self.theta_e) and support_met
        return {'C':C,'theta_e':self.theta_e,'above_threshold':C>self.theta_e,
                'support_channels':above_min,'k_required':self.k,'support_met':support_met,
                'encounter':encounter,'weights':W}

class HistoryUpdater:
    def eta(self, S, H):
        eigs = np.linalg.eigvalsh(H.A)
        anisotropy = float(-np.sum(eigs*np.log(eigs+1e-10)))
        eta_max = float(np.clip(0.5*np.exp(-0.3*anisotropy), 0.05, 0.5))
        return eta_max*(1.0-np.exp(-S/0.3))
    def update_H1(self, H, eps1, eta):
        delta = abs(eps1-H.mu1)/(H.sigma1+1e-10)-1.0
        H.mu1 = H.mu1+eta*(eps1-H.mu1)
        H.sigma1 = max(H.sigma1_min, H.sigma1+eta*delta*H.sigma1)
    def update_H2(self, H, eps2, r2_current, eta):
        u = (eps2 + 1.0) / 2.0
        novelty = 1.0 - r2_current
        target_alpha = u
        target_beta = 1.0 - u
        H.alpha = max(0.05, H.alpha + eta * novelty * (target_alpha - H.alpha))
        H.beta = max(0.01, H.beta + eta * novelty * (target_beta - H.beta))
    def update_H3(self, H, eps3, S, eta):
        eps3_unit=eps3/(np.linalg.norm(eps3)+1e-12); outer=np.outer(eps3_unit,eps3_unit)
        complement=np.eye(DIM)-outer; frob=float(np.linalg.norm(H.A,'fro'))
        rho=0.5*np.exp(-0.5*frob); lam_decay=0.1+0.4*min(S/0.5,1.0)
        dA=rho*(outer-lam_decay*complement@H.A@complement)
        H.A=H.A+eta*dA; H._project_A()
    def update_H4(self, H, eps4_rhythm, eta):
        m=len(H.B_base)
        eps4_m=eps4_rhythm[:m] if len(eps4_rhythm)>=m else np.pad(eps4_rhythm,(0,m-len(eps4_rhythm)))
        power=eps4_m**2; ps=np.sum(power)
        if ps < 1e-12: return
        pn=power/ps; exp=pn*(1.0-H.B_base); sup=0.1*H.B_base*(1.0-pn)
        H.B_base=np.maximum(H.B_base+eta*(exp-sup),1e-6)
        H.B_base=H.B_base/(np.sum(H.B_base)+1e-10)
    def update_H5(self, H, eps5, S_hold, eta):
        target=max(eps5,H.tau_hold_base)
        H.tau_hold_base=H.tau_hold_base+eta*S_hold*(target-H.tau_hold_base)
    def apply_all(self, H, press, r, S):
        eta_val=self.eta(S,H)
        self.update_H1(H,press.magnitude,eta_val); self.update_H2(H,press.valence,r['r2'],eta_val)
        self.update_H3(H,press.direction,S,eta_val); self.update_H4(H,press.rhythm,eta_val)
        self.update_H5(H,press.persistence,r['r1'],eta_val)
        return eta_val

class ExistencePressLibrary:
    def _make_rhythm(self, dominant_bin, spread=1.0, m=8):
        bins=np.arange(m,dtype=float); r=np.exp(-(bins-dominant_bin)**2/(2*spread**2))
        return r/(np.sum(r)+1e-10)
    def _dir(self, *weights):
        w=np.array(weights,dtype=float); return w/(np.linalg.norm(w)+1e-10)
    def water(self): return ExistencePress(0.5,0.3,self._dir(0.1,0.2,0.2,0.8,0.2),self._make_rhythm(1,2.0),2.0,"water","elemental")
    def breath(self): return ExistencePress(0.4,0.2,self._dir(0.9,0.1,0.1,0.2,0.1),self._make_rhythm(2,0.5),0.5,"breath","elemental")
    def gravity(self): return ExistencePress(0.3,0.0,self._dir(0.1,0.1,0.5,0.6,0.1),self._make_rhythm(0,3.0),10.0,"gravity","elemental")
    def light(self): return ExistencePress(0.8,0.4,self._dir(0.3,0.2,0.2,0.5,0.5),self._make_rhythm(5,1.5),1.0,"light","elemental")
    def fire_moderate(self): return ExistencePress(0.6,0.5,self._dir(0.1,0.8,0.3,0.2,0.1),self._make_rhythm(3,1.0),0.8,"fire_moderate","elemental")
    def fire_extreme(self): return ExistencePress(2.0,-0.8,self._dir(0.1,0.9,0.5,0.1,0.1),self._make_rhythm(6,0.5),0.3,"fire_extreme","elemental")
    def ground(self): return ExistencePress(0.4,0.1,self._dir(0.1,0.1,0.9,0.3,0.1),self._make_rhythm(0,4.0),10.0,"ground","elemental")
    def hunger(self): return ExistencePress(0.7,-0.4,self._dir(0.2,0.8,0.2,0.2,0.2),self._make_rhythm(1,2.0),3.0,"hunger","biological")
    def satiation(self): return ExistencePress(0.4,0.7,self._dir(0.1,0.3,0.1,0.8,0.2),self._make_rhythm(1,2.0),1.5,"satiation","biological")
    def pain_moderate(self): return ExistencePress(0.8,-0.7,self._dir(0.2,0.6,0.7,0.1,0.2),self._make_rhythm(5,0.8),0.5,"pain_moderate","biological")
    def pain_extreme(self): return ExistencePress(1.8,-1.0,self._dir(0.1,0.5,0.8,0.1,0.1),self._make_rhythm(7,0.5),0.2,"pain_extreme","biological")
    def fatigue(self): return ExistencePress(0.3,-0.2,self._dir(0.7,0.1,0.1,0.2,0.1),self._make_rhythm(0,3.0),2.0,"fatigue","biological")
    def caregiver_face(self): return ExistencePress(0.6,0.6,self._dir(0.5,0.3,0.5,0.6,0.4),self._make_rhythm(3,1.5),1.5,"caregiver_face","relational")
    def caregiver_voice(self): return ExistencePress(0.5,0.5,self._dir(0.7,0.2,0.2,0.3,0.3),self._make_rhythm(3,1.0),1.0,"caregiver_voice","relational")
    def absence(self): return ExistencePress(0.1,-0.6,self._dir(0.5,0.1,0.6,0.3,0.4),self._make_rhythm(0,4.0),2.0,"absence","relational")
    def threat(self): return ExistencePress(0.9,-0.8,self._dir(0.2,0.6,0.7,0.1,0.5),self._make_rhythm(6,0.5),0.4,"threat","relational")
    def prenatal_steady(self): return ExistencePress(0.3,0.5,self._dir(0.6,0.1,0.3,0.3,0.1),self._make_rhythm(2,0.3),5.0,"prenatal_steady","developmental")
    def birth_event(self): return ExistencePress(1.4,-0.3,self._dir(0.5,0.3,0.9,0.2,0.4),self._make_rhythm(6,1.0),0.8,"birth_event","developmental")
    def early_postnatal_skin(self): return ExistencePress(0.4,0.7,self._dir(0.2,0.2,0.8,0.4,0.1),self._make_rhythm(2,0.5),3.0,"early_postnatal_skin","developmental")
    def music_full(self): return ExistencePress(0.6,0.4,self._dir(0.6,0.5,0.4,0.6,0.7),self._make_rhythm(3,2.0),2.0,"music_full","conceptual")
    def mathematics(self): return ExistencePress(0.4,0.3,self._dir(0.1,0.2,0.1,0.9,0.3),self._make_rhythm(0,4.0),5.0,"mathematics","conceptual")
    def death_encounter(self): return ExistencePress(1.0,-0.5,self._dir(0.6,0.1,0.6,0.4,0.9),self._make_rhythm(0,4.0),0.1,"death_encounter","conceptual")
    def get_press(self, name):
        m = {n: getattr(self,n) for n in self.list_all()}
        fn = m.get(name); return fn() if fn else None
    def list_all(self):
        return ['water','breath','gravity','light','fire_moderate','fire_extreme','ground',
                'hunger','satiation','pain_moderate','pain_extreme','fatigue',
                'caregiver_face','caregiver_voice','absence','threat',
                'prenatal_steady','birth_event','early_postnatal_skin',
                'music_full','mathematics','death_encounter']

class ConditionalEncounterEnvironment:
    def __init__(self, seed=0):
        np.random.seed(seed)
        self.H = ConditionFieldState(); self.Z = ConsolidationState()
        self.h = np.zeros(DIM); self.q = 1.0; self.B_fatigue = np.zeros(8)
        self.F = ActiveConditionField(); self.R = ReceivabilityOperators()
        self.gate = CommensurabilityGate(); self.updater = HistoryUpdater()
        self.t = 0; self.press_history = []; self.encounter_history = []
        self.no_encounter_history = []; self.Sigma = []

    def receive(self, press, h_perturbation=None):
        self.t += 1
        self.h = self.h*0.9 + (h_perturbation*0.1 if h_perturbation is not None else 0)
        self.q = min(1.0, self.q+0.02)
        lam = self.F.compute(self.H, self.h, self.B_fatigue, self.q)
        r = self.R.compute_all(press, lam)
        C_result = self.gate.compute(r, lam['A_act'])
        if press.magnitude > 1.5:
            C_result = dict(C_result)
            C_result['encounter'] = False
            C_result['magnitude_overwhelm'] = True
            C_result['overwhelm_magnitude'] = float(press.magnitude)
        result = {'t':self.t,'press':press.label,'press_class':press.press_class,
                  'epsilon':{'magnitude':round(press.magnitude,4),'valence':round(press.valence,4),
                             'direction':press.direction.round(3).tolist(),'persistence':round(press.persistence,4)},
                  'receivability':{'r1_magnitude':round(r['r1'],4),'r2_valence':round(r['r2'],4),
                                   'r3_direction':round(r['r3'],4),'r4_rhythm':round(r['r4'],4),
                                   'r5_persistence':round(r['r5'],4),'sigma_profile':r['sigma_profile'].round(4).tolist()},
                  'commensurability':{'C':round(C_result['C'],4),'theta_e':C_result['theta_e'],
                                      'above_threshold':C_result['above_threshold'],
                                      'support_channels':C_result['support_channels'],
                                      'support_met':C_result['support_met'],'encounter':C_result['encounter']},
                  'condition_field':{'mu1':round(lam['mu1_act'],4),'sigma1':round(lam['sigma1_act'],4),
                                     'alpha':round(lam['alpha_act'],4),'beta':round(lam['beta_act'],4),
                                     'tau_hold_act':round(lam['tau_hold_act'],4)}}
        if C_result['encounter']:
            m = len(self.B_fatigue)
            self.B_fatigue = self.B_fatigue*0.8 + r['sigma_profile'][3]*press.rhythm[:m]*0.2
            self.q = max(0.1, self.q-0.05)
            S = float(np.linalg.norm(r['sigma_profile']))
            eta_val = self.updater.apply_all(self.H, press, r, S)
            delta_eps = r['sigma_profile']*np.array([press.magnitude,(press.valence+1)/2,1.0,float(np.max(press.rhythm)),press.persistence])
            self.Sigma.append({'t':self.t,'press':press.label,'delta_epsilon':delta_eps.round(4).tolist(),'S':round(S,4)})
            result['encounter_result'] = {'occurred':True,'significance':round(S,4),'eta':round(eta_val,4),'delta_epsilon':delta_eps.round(4).tolist(),'H_unsettledness':round(self._unsettledness(),4)}
            self.encounter_history.append(result)
        else:
            result['encounter_result'] = {'occurred':False,'reason':self._fail_reason(C_result,r)}
            self.no_encounter_history.append(result)
        self.press_history.append(result)
        return result

    def _unsettledness(self):
        diff = (self.H.mu1-self.Z.mu1)**2+(self.H.sigma1-self.Z.sigma1)**2+(self.H.alpha-self.Z.alpha)**2+(self.H.beta-self.Z.beta)**2+float(np.sum((self.H.A-self.Z.A)**2))
        return float(np.sqrt(diff))

    def _fail_reason(self, C_result, r):
        if C_result.get('magnitude_overwhelm'):
            m = C_result.get('overwhelm_magnitude', 0.0)
            return f"magnitude_overwhelm: press.magnitude={m:.3f} above R1 ceiling 1.5"
        if not C_result['above_threshold']: return f"C(t)={C_result['C']:.3f} below theta_e={C_result['theta_e']}"
        if not C_result['support_met']: return f"only {C_result['support_channels']}/{C_result['k_required']} channels above floor"
        return "unknown"

    def self_observe(self):
        eigvals, eigvecs = np.linalg.eigh(self.H.A)
        dominant_idx = np.argmax(eigvals); dominant_dir = eigvecs[:,dominant_idx]
        internal_press = ExistencePress(magnitude=self.H.mu1,
            valence=(self.H.alpha-self.H.beta)/(self.H.alpha+self.H.beta+1e-10),
            direction=dominant_dir, rhythm=self.H.B_base.copy(),
            persistence=self.H.tau_hold_base, label="self_observation", press_class="internal")
        result = self.receive(internal_press)
        names = ['temporal','agency','boundary','reality','exit']
        result['self_observation'] = {'dominant_eigenmode':names[dominant_idx],'dominant_eigenvalue':round(float(eigvals[dominant_idx]),4),
            'min_eigenvalue':round(float(eigvals[0]),4),'blind_direction':names[np.argmin(eigvals)],'alpha_beta_ratio':round(float(self.H.alpha/(self.H.beta+1e-10)),3)}
        return result

    def report_state(self):
        lines = ["="*56,"CONDITION FIELD STATE","="*56,f"Time: {self.t}",
                 f"Encounters: {len(self.encounter_history)} / {len(self.press_history)}",
                 f"Sigma: {len(self.Sigma)} entries",f"Unsettledness: {self._unsettledness():.4f}","",
                 f"H1 mu1={self.H.mu1:.3f} sigma1={self.H.sigma1:.3f}",
                 f"H2 alpha={self.H.alpha:.3f} beta={self.H.beta:.3f}"]
        eigvals = np.linalg.eigvalsh(self.H.A)
        names = ['temporal','agency','boundary','reality','exit']
        for v,n in zip(eigvals[::-1],names): lines.append(f"  {n:10s}: {v:.4f}")
        lines.append("="*56)
        return "\n".join(lines)

if __name__ == "__main__":
    lib = ExistencePressLibrary(); env = ConditionalEncounterEnvironment(seed=42)
    for _ in range(20): env.receive(lib.prenatal_steady())
    print(f"After 20 prenatal: {len(env.encounter_history)} encounters")
    r = env.receive(lib.birth_event()); print(f"Birth: encountered={r['encounter_result']['occurred']}")
    print(env.report_state())
    print("Conditional encounter environment operational.")
