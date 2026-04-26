"""
EFT VERIFIED CORE - Source-verified equations
The Mirror Platform LLC - Ilya Belous - April 2026
"""

import numpy as np
from scipy.linalg import expm, logm
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')

DIM = 5
GHOST_FLOOR = 0.05
DEPOSIT_THRESHOLD = 0.12
LAMBDA_DECAY = 15.0
KAPPA = 0.3
BETA = 1.0
SIGMA_MIN = 0.3
SIGMA_MAX = 1.5
S_THRESH = 0.15
T_SIG = 0.1
B_DIM = 3
H_DIM = DIM
GAMMA_BODY = 0.1
ALIGNMENT_REGEN = 0.8

EIGENMODE_DIRS = np.eye(DIM)
V_T, V_A, V_B, V_R, V_E = 0, 1, 2, 3, 4
D4_INDICES = [V_T, V_A, V_B, V_R]
EXIT_INDEX = V_E

class BodySubstrate:
    def __init__(self, seed=0):
        np.random.seed(seed)
        self.b_star = np.zeros(B_DIM)
        self.b = self.b_star.copy()
        self.h_0 = np.zeros(H_DIM)
        self.W_phi = np.array([
            [0.1, 0.6, 0.4, 0.2, 0.1],
            [0.3, 0.3, 0.3, 0.3, 0.3],
            [0.5, 0.1, 0.1, 0.2, 0.6],
        ])
        self.time = 0

    def phi(self, b):
        return self.h_0 + self.W_phi.T @ (b - self.b_star)

    def step(self, encounter_perturbation=None, dt=0.1):
        restoration = -GAMMA_BODY * (self.b - self.b_star)
        xi = encounter_perturbation if encounter_perturbation is not None else np.zeros(B_DIM)
        self.b = self.b + dt * (restoration + xi)
        self.time += dt
        return self.phi(self.b)

    def h(self):
        return self.phi(self.b)

    def G_field(self):
        h_current = self.h()
        return expm(np.diag(h_current))

    def G_body(self):
        b_deviation = self.b - self.b_star
        phi_body = self.W_phi.T @ b_deviation
        return expm(np.diag(phi_body * 0.1))

def sigma_from_salience(s):
    sigmoid_val = 1.0 / (1.0 + np.exp(-(s - S_THRESH) / T_SIG))
    return SIGMA_MIN + (SIGMA_MAX - SIGMA_MIN) * sigmoid_val

@dataclass
class Trace:
    position: np.ndarray
    significance: float
    kernel_width: float
    normal: np.ndarray
    deposit_time: int
    current_age: int = 0

    def persistence(self):
        return np.exp(-self.current_age / LAMBDA_DECAY)

    def effective_significance(self):
        return max(self.significance * self.persistence(), GHOST_FLOOR)

    def kernel_at(self, x):
        return float(np.exp(-np.dot(x - self.position, x - self.position) /
                            (2 * self.kernel_width ** 2)))

def build_S_traces(traces, x):
    S = np.zeros((DIM, DIM))
    for trace in traces:
        k = trace.kernel_at(x)
        s_eff = trace.effective_significance()
        outer = np.outer(trace.normal, trace.normal)
        S -= s_eff * k * outer
    return S

def build_G_total(traces, x, G_field, G_body, w_f=0.3, w_b=0.1, w_t=0.6):
    S = build_S_traces(traces, x)
    try:
        log_G_field = logm(G_field)
    except Exception:
        log_G_field = np.zeros((DIM, DIM))
    try:
        log_G_body = logm(G_body)
    except Exception:
        log_G_body = np.zeros((DIM, DIM))
    exponent = w_f * np.real(log_G_field) + w_b * np.real(log_G_body) + w_t * S
    G = expm(exponent)
    G = np.real(G)
    eigvals = np.linalg.eigvalsh(G)
    if np.min(eigvals) < 1e-6:
        G += (1e-6 - np.min(eigvals) + 1e-8) * np.eye(DIM)
    return G

def Psi(x, traces, G_field, G_body):
    G = build_G_total(traces, x, G_field, G_body)
    sign, logdet = np.linalg.slogdet(G)
    if sign <= 0:
        return 1e10
    return -float(logdet)

def find_R_star(traces, G_field, G_body, n_steps=100, lr=0.02):
    if not traces:
        return np.zeros(DIM)
    positions = np.array([t.position for t in traces])
    weights = np.array([t.effective_significance() for t in traces])
    weights = weights / (weights.sum() + 1e-12)
    x = np.sum(weights[:, None] * positions, axis=0)
    eps = 1e-4
    prev_psi = Psi(x, traces, G_field, G_body)
    for _ in range(n_steps):
        grad = np.zeros(DIM)
        for i in range(DIM):
            xp = x.copy(); xp[i] += eps
            xm = x.copy(); xm[i] -= eps
            grad[i] = (Psi(xp, traces, G_field, G_body) -
                       Psi(xm, traces, G_field, G_body)) / (2 * eps)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-6:
            break
        step = lr / (1.0 + grad_norm)
        x_new = x - step * grad
        new_psi = Psi(x_new, traces, G_field, G_body)
        if new_psi < prev_psi:
            x = x_new
            prev_psi = new_psi
        else:
            lr *= 0.5
    return x

def compute_Hessian_at_R_star(R_star, traces, G_field, G_body, eps=0.05):
    H = np.zeros((DIM, DIM))
    for i in range(DIM):
        for j in range(DIM):
            xpp = R_star.copy(); xpp[i] += eps; xpp[j] += eps
            xpm = R_star.copy(); xpm[i] += eps; xpm[j] -= eps
            xmp = R_star.copy(); xmp[i] -= eps; xmp[j] += eps
            xmm = R_star.copy(); xmm[i] -= eps; xmm[j] -= eps
            H[i,j] = (Psi(xpp, traces, G_field, G_body) -
                      Psi(xpm, traces, G_field, G_body) -
                      Psi(xmp, traces, G_field, G_body) +
                      Psi(xmm, traces, G_field, G_body)) / (4 * eps * eps)
    return H

class SignificanceFunctional:
    def __init__(self):
        self.baseline_I_SB = None

    def component1_delta_G(self, G_before, G_after):
        delta_G = G_after - G_before
        frob = float(np.linalg.norm(delta_G, 'fro'))
        if frob < 1e-12:
            return 0.0, 0.0
        try:
            s_vals = np.linalg.svd(delta_G, compute_uv=False)
            spatial_conc = float(s_vals[0]) / (frob + 1e-12)
        except Exception:
            spatial_conc = 1.0 / np.sqrt(DIM)
        return frob * spatial_conc, spatial_conc

    def component3_I_SB(self, x, traces, G_field, G_body,
                        noise_amp=0.1, n_probes=12, update_baseline=True):
        divergences = []
        G_clean = build_G_total(traces, x, G_field, G_body)
        for _ in range(n_probes):
            perturbed = x + np.random.randn(DIM) * noise_amp
            G_pert = build_G_total(traces, perturbed, G_field, G_body)
            divergences.append(float(np.linalg.norm(G_pert - G_clean, 'fro')))
        mean_div = float(np.mean(divergences)) + 1e-8
        I_SB_raw = 1.0 / mean_div
        if self.baseline_I_SB is None and update_baseline:
            self.baseline_I_SB = I_SB_raw
        if self.baseline_I_SB is None:
            return 1.0
        return I_SB_raw / (self.baseline_I_SB + 1e-8)

    def compute(self, G_before, G_after, x, traces, G_field, G_body, tau=0):
        dG_norm, spatial_conc = self.component1_delta_G(G_before, G_after)
        P_tau = float(np.exp(-tau / LAMBDA_DECAY))
        I_SB = self.component3_I_SB(x, traces, G_field, G_body, update_baseline=False)
        if self.baseline_I_SB is None:
            self.baseline_I_SB = 1.0 / (1e-8 + 0.01)
            I_SB = 1.0
        s = dG_norm * P_tau * I_SB
        return {
            "delta_G_norm": dG_norm,
            "spatial_concentration": spatial_conc,
            "P_tau_lambda": P_tau,
            "I_SB": I_SB,
            "significance": s,
            "above_ghost_floor": s >= GHOST_FLOOR,
            "above_deposit_threshold": s >= DEPOSIT_THRESHOLD,
        }

class ConflictOperator:
    def __init__(self, kappa=KAPPA):
        self.kappa = kappa

    def apply(self, T, G):
        G_dev = G - np.eye(DIM)
        K_sym = self.kappa * (G_dev + G_dev.T) / 2.0
        return K_sym @ T

class EvolutionRateTensor:
    def compute(self, s_enc, x_p, p_enc, n_enc, n_p,
                h_t, h_0, w_domain_p, rho_p, Hess_Psi_np,
                tau_p, lambda_p, f_0=1.0):
        sigma = sigma_from_salience(s_enc)
        K_val = float(np.exp(-np.dot(x_p - p_enc, x_p - p_enc) / (2 * sigma**2)))
        alignment_sq = float(np.dot(n_enc, n_p))**2
        A_p = s_enc * K_val * alignment_sq
        n_p_index = int(np.argmax(np.abs(n_p)))
        n_enc_index = int(np.argmax(np.abs(n_enc)))
        exit_blocked = (n_p_index == EXIT_INDEX) and (n_enc_index in D4_INDICES)
        if exit_blocked:
            A_p = 0.0
        domain_projection = float(np.dot(w_domain_p, h_t - h_0))
        F_p = f_0 * float(np.exp(domain_projection))
        R_p = 1.0 + rho_p * max(Hess_Psi_np, 0.0)
        D_p = max(float(np.exp(-tau_p / lambda_p)), GHOST_FLOOR)
        E_p = A_p * F_p / R_p * D_p
        return {
            "E_p": E_p, "A_p_alignment": A_p, "F_p_field": F_p,
            "R_p_resistance": R_p, "D_p_decay": D_p,
            "exit_blocked": exit_blocked, "sigma": sigma,
            "alignment_sq": alignment_sq,
        }

class TauIrrTracker:
    def __init__(self, H_min=1.0, A_threshold=5.0):
        self.H_min = H_min
        self.A_threshold = A_threshold
        self.A_accumulated: Dict[str, float] = {}
        self.status: Dict[str, str] = {}
        self.dependents: Dict[str, List[str]] = {}

    def update(self, part_id, E_p, dt=1.0):
        if part_id not in self.A_accumulated:
            self.A_accumulated[part_id] = 0.0
            self.status[part_id] = "PRE-ENCOUNTER"
        self.A_accumulated[part_id] += E_p * dt
        A = self.A_accumulated[part_id]
        if A < self.H_min:
            self.status[part_id] = "PRE-ENCOUNTER"
        else:
            self.status[part_id] = "UNIQUE"

    def register_dependent(self, parent_id, dependent_id):
        if parent_id not in self.dependents:
            self.dependents[parent_id] = []
        self.dependents[parent_id].append(dependent_id)

print("EFT Verified Core loaded.")
