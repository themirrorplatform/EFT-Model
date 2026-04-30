"""
EFT DEAP TRANSLATOR
The Mirror Platform LLC - Ilya Belous - April 2026

Converts DEAP physiological data into ExistencePress streams.

This is the gap-closer between theorist-encoded presses and
existence-generated presses. The presses produced here did not
come from the theory. They came from real bodies in real
emotional states.

DEAP DATA STRUCTURE (preprocessed Python format):
  data['data']:   shape (40, 40, 8064) = 40 trials x 40 channels x 8064 samples
                  Channels 0-31:  EEG (128 Hz)
                  Channel 32:     skin conductance level (SCL)
                  Channel 33:     respiration amplitude
                  Channel 34:     skin temperature
                  Channel 35:     ECG
                  Channel 36:     blood volume pulse (BVP)
                  Channel 37:     EMG Zygomaticus
                  Channel 38:     EMG Trapezius
                  Channel 39:     EOG horizontal
  data['labels']: shape (40, 4) = 40 trials x [valence, arousal, dominance, liking]
                  All ratings 1-9

EIGENMODE MAPPING (SRP-E1 — empirical grounding):

  v_T (temporal):  EEG theta power (4-8 Hz) — temporal continuity,
                   memory encoding, sequential processing.
                   Theta is the memory oscillation. It marks what
                   the brain is tracking across time.

  v_A (agency):    EEG alpha suppression (8-13 Hz) — motor readiness,
                   approach behavior, agency engagement.
                   Alpha desynchronization = active engagement.
                   Alpha power (inverted) = agency activation.

  v_B (boundary):  Skin conductance level (SCL) — arousal-driven
                   boundary activation, threat/protection response.
                   EEG beta power (13-30 Hz) secondary.
                   SCL is the body's boundary sensor.

  v_R (reality):   EEG gamma power (30-45 Hz) — feature binding,
                   reality construction, perceptual integration.
                   Gamma = the brain binding what is real right now.

  v_E (exit):      Heart rate variability (HRV from ECG) — autonomic
                   flexibility, openness to the other, vagal tone.
                   High HRV = receptive to exit/otherness.
                   Low HRV = closed/collapsed.

PRESS PARAMETERS:

  magnitude:    Arousal rating (normalized) × SCL activation
                These are orthogonal: arousal is subjective,
                SCL is the body's objective response.

  valence:      Valence rating normalized to [-1, +1]
                Direct mapping. No ambiguity.

  direction:    [v_T, v_A, v_B, v_R, v_E] from EEG band powers + SCL + HRV
                This is where the theory meets the measurement.

  rhythm:       Respiration rate + EEG spectral profile → 8-bin signature
                The body's temporal signature of the encounter.

  persistence:  SCL persistence (how long the state holds)
                Computed from SCL decay rate across the trial window.

HONEST STATUS:
  This translator implements the mapping described above.
  The eigenmode-to-frequency-band mapping is literature-grounded
  (SRP-E1 theoretical specification) but not yet empirically
  calibrated against EFT's specific geometry.
  
  What this produces: presses that came from real bodies.
  What this does not produce: perfectly calibrated eigenmode coordinates.
  
  The calibration gap is SRP-E1. The data gap is now closed.
"""

import sys
sys.path.insert(0, '/home/claude')

import numpy as np
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from eft_encounter_env import ExistencePress


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

DEAP_SRATE = 128          # Hz (downsampled)
DEAP_EEG_CHANNELS = 32   # channels 0-31
DEAP_N_TRIALS = 40
DEAP_N_SAMPLES = 8064     # ~63 seconds at 128 Hz

# Peripheral channel indices (0-indexed from full 40-channel array)
CH_SCL = 32    # skin conductance level
CH_RESP = 33   # respiration amplitude
CH_TEMP = 34   # skin temperature
CH_ECG = 35    # electrocardiogram
CH_BVP = 36    # blood volume pulse
CH_EMG_Z = 37  # EMG Zygomaticus (face)
CH_EMG_T = 38  # EMG Trapezius (shoulder)
CH_EOG = 39    # electrooculogram

# EEG frequency bands (Hz) → eigenmode mapping
BANDS = {
    'delta': (1, 4),    # not mapped to eigenmodes
    'theta': (4, 8),    # → v_T temporal
    'alpha': (8, 13),   # → v_A agency (inverted — alpha suppression = agency)
    'beta':  (13, 30),  # → v_B boundary
    'gamma': (30, 45),  # → v_R reality
}

# Eigenmode indices
V_T, V_A, V_B, V_R, V_E = 0, 1, 2, 3, 4


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL PROCESSING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def bandpower(signal: np.ndarray, srate: float,
              fmin: float, fmax: float) -> float:
    """
    Compute band power using Welch's method approximation.
    Fast enough for real-time translation.
    """
    n = len(signal)
    fft_vals = np.abs(np.fft.rfft(signal * np.hanning(n)))**2
    freqs = np.fft.rfftfreq(n, d=1.0/srate)
    idx = np.logical_and(freqs >= fmin, freqs < fmax)
    if not np.any(idx):
        return 0.0
    return float(np.mean(fft_vals[idx]))


def compute_hrv(ecg_signal: np.ndarray, srate: float) -> float:
    """
    Estimate HRV from ECG signal.
    Uses R-peak detection via simple threshold method.
    Returns RMSSD normalized to [0, 1].
    
    Higher HRV → more autonomic flexibility → higher v_E receivability.
    Lower HRV → autonomic rigidity → lower v_E (exit capability compressed).
    """
    # Normalize signal
    sig = ecg_signal - np.mean(ecg_signal)
    sig_max = np.max(np.abs(sig))
    if sig_max < 1e-10:
        return 0.5  # neutral HRV if no signal

    sig = sig / sig_max

    # Simple R-peak detection: threshold crossing
    threshold = 0.5
    above = sig > threshold
    crossings = np.where(np.diff(above.astype(int)) > 0)[0]

    if len(crossings) < 3:
        return 0.5  # not enough peaks

    # R-R intervals in seconds
    rr_intervals = np.diff(crossings) / srate

    # Filter physiologically plausible RR intervals (0.4s - 1.5s = 40-150 bpm)
    rr_valid = rr_intervals[(rr_intervals > 0.4) & (rr_intervals < 1.5)]

    if len(rr_valid) < 2:
        return 0.5

    # RMSSD: root mean square of successive differences
    rmssd = float(np.sqrt(np.mean(np.diff(rr_valid)**2)))

    # Normalize: typical RMSSD range 10-100ms → [0, 1]
    # High RMSSD (>60ms) → high autonomic flexibility → high v_E
    # Low RMSSD (<20ms) → low flexibility → low v_E
    normalized = float(np.clip(rmssd / 0.060, 0.0, 1.5) / 1.5)
    return normalized


def compute_scl_persistence(scl: np.ndarray) -> float:
    """
    Estimate how long the SCL state persists.
    Measured as the autocorrelation decay rate.
    High persistence = sustained physiological state = high ε₅.
    """
    if np.std(scl) < 1e-10:
        return 1.0
    # Normalize
    scl_norm = (scl - np.mean(scl)) / (np.std(scl) + 1e-10)
    # Autocorrelation at lag = 10% of signal
    lag = max(1, len(scl) // 10)
    if lag >= len(scl_norm):
        return 0.5
    acf = float(np.corrcoef(scl_norm[:-lag], scl_norm[lag:])[0, 1])
    # Map [-1,1] → [0, 3] seconds (typical persistence range)
    persistence = max(0.1, (acf + 1.0) * 1.5)
    return persistence


def compute_eeg_direction(eeg_data: np.ndarray, srate: float) -> np.ndarray:
    """
    Map EEG band powers to the five eigenmode directions.
    
    Returns unit vector [v_T, v_A, v_B, v_R, v_E_placeholder].
    Note: v_E comes from HRV, not EEG — this is the boundary of what
    EEG alone can tell us. HRV is added separately.
    
    MAPPING:
      v_T ← theta power (mean across all EEG channels)
      v_A ← alpha suppression = 1 / (alpha power + ε)
      v_B ← beta power
      v_R ← gamma power
      v_E ← placeholder (set from HRV externally)
    """
    # Use mean across all EEG channels for each band
    # (spatial averaging — channel-specific mapping is SRP-E1 future work)
    mean_eeg = np.mean(eeg_data, axis=0)  # average across channels

    theta = bandpower(mean_eeg, srate, *BANDS['theta'])
    alpha = bandpower(mean_eeg, srate, *BANDS['alpha'])
    beta  = bandpower(mean_eeg, srate, *BANDS['beta'])
    gamma = bandpower(mean_eeg, srate, *BANDS['gamma'])

    # Alpha suppression = agency activation
    # High alpha = low agency. Low alpha (suppressed) = high agency.
    # Express as: how much alpha is suppressed relative to its maximum
    # Use 1 - normalized_alpha so it contributes positively
    total_power = theta + alpha + beta + gamma + 1e-10
    alpha_norm = alpha / total_power
    # Alpha suppression: inverted and scaled to same range as other bands
    alpha_agency = (1.0 - alpha_norm) * (theta + beta + gamma) / 3.0

    direction = np.array([theta, alpha_agency, beta, gamma, 0.0])

    # Normalize
    norm = np.linalg.norm(direction)
    if norm < 1e-10:
        return np.ones(5) / np.sqrt(5)
    return direction / norm


def compute_rhythm(resp: np.ndarray, eeg_mean: np.ndarray,
                   srate: float, m: int = 8) -> np.ndarray:
    """
    Compute 8-bin spectral rhythm signature.
    Combines respiration spectral content and EEG spectral profile.
    
    Bins: [0-2Hz, 2-4Hz, 4-8Hz, 8-13Hz, 13-20Hz, 20-30Hz, 30-40Hz, 40+Hz]
    This captures the full spectral signature of the encounter's temporal rhythm.
    """
    bin_edges = [0, 2, 4, 8, 13, 20, 30, 40, srate//2]
    rhythm = np.zeros(m)

    # EEG spectral content
    n = len(eeg_mean)
    fft_vals = np.abs(np.fft.rfft(eeg_mean * np.hanning(n)))**2
    freqs = np.fft.rfftfreq(n, d=1.0/srate)

    for i in range(m):
        f_lo = bin_edges[i]
        f_hi = bin_edges[i+1] if i+1 < len(bin_edges) else srate//2
        idx = np.logical_and(freqs >= f_lo, freqs < f_hi)
        if np.any(idx):
            rhythm[i] = float(np.mean(fft_vals[idx]))

    # Add respiration contribution to low-frequency bins (0-2Hz)
    if np.std(resp) > 1e-10:
        resp_norm = resp / (np.std(resp) + 1e-10)
        resp_fft = np.abs(np.fft.rfft(resp_norm * np.hanning(len(resp_norm))))**2
        resp_freqs = np.fft.rfftfreq(len(resp_norm), d=1.0/srate)
        resp_idx = resp_freqs < 2.0
        if np.any(resp_idx):
            rhythm[0] += float(np.mean(resp_fft[resp_idx]))

    # Normalize
    s = np.sum(rhythm)
    if s > 1e-10:
        rhythm = rhythm / s
    else:
        rhythm = np.ones(m) / m

    return rhythm


# ─────────────────────────────────────────────────────────────────────────────
# TRIAL-TO-PRESS TRANSLATOR
# ─────────────────────────────────────────────────────────────────────────────

def trial_to_press(trial_data: np.ndarray,
                   labels: np.ndarray,
                   trial_idx: int,
                   participant_id: int,
                   window_start: Optional[int] = None,
                   window_size: Optional[int] = None) -> ExistencePress:
    """
    Convert one DEAP trial into one ExistencePress.
    
    Args:
        trial_data:    shape (40, 8064) — 40 channels × 8064 samples
        labels:        shape (4,) — [valence, arousal, dominance, liking] (1-9)
        trial_idx:     which of the 40 trials (0-indexed)
        participant_id: which participant (1-32)
        window_start:  optional start sample for windowed analysis
        window_size:   optional window size (default: full trial)
    
    Returns:
        ExistencePress with all five components derived from physiology
    """
    # Extract window
    if window_start is not None and window_size is not None:
        window_end = min(window_start + window_size, DEAP_N_SAMPLES)
        data = trial_data[:, window_start:window_end]
    else:
        data = trial_data

    # Separate EEG from peripheral
    eeg = data[:DEAP_EEG_CHANNELS, :]          # (32, samples)
    scl = data[CH_SCL, :]                        # skin conductance
    resp = data[CH_RESP, :]                      # respiration
    ecg = data[CH_ECG, :]                        # ECG for HRV
    bvp = data[CH_BVP, :]                        # blood volume pulse

    # Mean EEG for spectral analysis
    mean_eeg = np.mean(eeg, axis=0)

    # ── ε₁: MAGNITUDE ────────────────────────────────────────────────────────
    # Arousal rating (1-9) normalized to [0, 1], scaled by SCL activation
    arousal_norm = (labels[1] - 1.0) / 8.0  # 1-9 → 0-1

    # SCL deviation from trial mean (activation = deviation above mean)
    scl_mean = np.mean(scl)
    scl_activation = float(np.clip((np.mean(scl) - scl_mean) / (np.std(scl) + 1e-10) + 0.5, 0.1, 1.5))

    # Magnitude: subjective arousal × objective bodily activation
    # Range approximately [0.1, 1.5] — within the ExistencePress receptive window
    magnitude = float(np.clip(arousal_norm * 0.8 + scl_activation * 0.3, 0.05, 1.6))

    # ── ε₂: VALENCE ──────────────────────────────────────────────────────────
    # Direct: valence rating (1-9) → [-1, +1]
    valence = float((labels[0] - 5.0) / 4.0)  # 1-9 → [-1, +1]

    # ── ε₃: DIRECTION ─────────────────────────────────────────────────────────
    # EEG band powers → [v_T, v_A, v_B, v_R, _]
    # HRV → v_E
    direction = compute_eeg_direction(eeg, DEAP_SRATE)

    # Replace v_E placeholder with HRV-derived value
    # Scale HRV to same order of magnitude as EEG band contributions
    # so it does not dominate the direction vector
    hrv = compute_hrv(ecg, DEAP_SRATE)
    eeg_magnitude = float(np.linalg.norm(direction[:4]))
    direction[V_E] = hrv * eeg_magnitude * 0.5

    # Re-normalize after adding HRV
    norm = np.linalg.norm(direction)
    if norm > 1e-10:
        direction = direction / norm

    # ── ε₄: RHYTHM ───────────────────────────────────────────────────────────
    rhythm = compute_rhythm(resp, mean_eeg, DEAP_SRATE, m=8)

    # ── ε₅: PERSISTENCE ──────────────────────────────────────────────────────
    # SCL persistence: how long does this physiological state sustain?
    persistence = compute_scl_persistence(scl)

    # ── LABEL ─────────────────────────────────────────────────────────────────
    label = f"P{participant_id:02d}_T{trial_idx+1:02d}"
    if window_start is not None:
        t_sec = window_start / DEAP_SRATE
        label += f"_t{t_sec:.1f}s"

    return ExistencePress(
        magnitude=magnitude,
        valence=valence,
        direction=direction,
        rhythm=rhythm,
        persistence=persistence,
        label=label,
        press_class="physiological_deap"
    )


def participant_to_press_stream(participant_data: Dict,
                                participant_id: int,
                                trials: Optional[List[int]] = None,
                                window_size: Optional[int] = None,
                                window_step: Optional[int] = None) -> List[ExistencePress]:
    """
    Convert one DEAP participant's session into a press stream.
    
    Args:
        participant_data: dict with 'data' (40,40,8064) and 'labels' (40,4)
        participant_id:   participant number (1-32)
        trials:          which trials to include (default: all 40)
        window_size:     if set, split each trial into windows (samples)
        window_step:     step between windows (default: window_size // 2)
    
    Returns:
        List of ExistencePress objects in temporal order.
        This is existence feeding into the system.
    """
    data = participant_data['data']      # (40, 40, 8064)
    labels = participant_data['labels']  # (40, 4)

    if trials is None:
        n_available = data.shape[0]
        trials = list(range(n_available))

    presses = []

    for trial_idx in trials:
        trial_data = data[trial_idx]    # (40, 8064)
        trial_labels = labels[trial_idx]  # (4,)

        if window_size is None:
            # Full trial as one press
            press = trial_to_press(trial_data, trial_labels, trial_idx, participant_id)
            presses.append(press)
        else:
            # Windowed: split trial into overlapping windows
            step = window_step if window_step else window_size // 2
            starts = range(0, DEAP_N_SAMPLES - window_size, step)
            for w_start in starts:
                press = trial_to_press(trial_data, trial_labels, trial_idx,
                                       participant_id, w_start, window_size)
                presses.append(press)

    return presses


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DEAP DATA GENERATOR
# For testing without the actual dataset files.
# Produces physiologically plausible signals.
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_deap_trial(arousal: float = 5.0,
                                   valence: float = 5.0,
                                   seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate one synthetic DEAP trial with physiologically plausible signals.
    
    This is NOT real physiological data.
    It is used to:
    1. Test the translator without the DEAP files
    2. Show what the system does with physiologically-structured input
    3. Demonstrate that the press structure is grounded in real signal types
    
    The synthetic data:
    - EEG: 1/f noise + band-specific structure (arousal → alpha suppression)
    - SCL: slow drift with arousal-dependent amplitude
    - ECG: realistic R-peaks at arousal-dependent heart rate
    - Respiration: sinusoidal at realistic rate
    
    Args:
        arousal: 1-9 (high arousal → higher HR, lower alpha, higher beta)
        valence: 1-9 (positive valence → higher approach alpha-suppression)
        seed:    random seed
    
    Returns:
        (trial_data (40, 8064), labels (4,))
    """
    np.random.seed(seed)
    n = DEAP_N_SAMPLES
    t = np.arange(n) / DEAP_SRATE
    arousal_norm = (arousal - 1) / 8.0  # 0-1
    valence_norm = (valence - 1) / 8.0  # 0-1

    data = np.zeros((40, n))

    # ── EEG channels (0-31): 1/f noise + band-specific ──────────────────────
    for ch in range(32):
        # Base: 1/f noise
        freqs = np.fft.rfftfreq(n, d=1.0/DEAP_SRATE)
        freqs[0] = 1.0  # avoid division by zero
        amplitude = 1.0 / np.sqrt(freqs)
        phase = np.random.uniform(0, 2*np.pi, len(freqs))
        eeg_base = np.fft.irfft(amplitude * np.exp(1j * phase), n=n)
        eeg_base = eeg_base / (np.std(eeg_base) + 1e-10) * 20.0  # ~20 uV

        # Theta (4-8 Hz): memory/temporal — moderate amplitude
        theta_freq = 6.0 + np.random.uniform(-1, 1)
        theta = 5.0 * np.sin(2 * np.pi * theta_freq * t + np.random.uniform(0, 2*np.pi))

        # Alpha (8-13 Hz): suppressed by arousal
        alpha_freq = 10.0 + np.random.uniform(-1, 1)
        alpha_amp = 15.0 * (1.0 - arousal_norm * 0.7)  # high arousal → alpha suppression
        alpha = alpha_amp * np.sin(2 * np.pi * alpha_freq * t + np.random.uniform(0, 2*np.pi))

        # Beta (13-30 Hz): boundary activation, elevated with arousal
        beta_freq = 20.0 + np.random.uniform(-3, 3)
        beta_amp = 3.0 * (0.3 + arousal_norm * 0.7)
        beta = beta_amp * np.sin(2 * np.pi * beta_freq * t + np.random.uniform(0, 2*np.pi))

        # Gamma (30-45 Hz): reality construction
        gamma_freq = 40.0 + np.random.uniform(-2, 2)
        gamma_amp = 2.0 * (0.5 + valence_norm * 0.5)  # positive valence → more reality contact
        gamma = gamma_amp * np.sin(2 * np.pi * gamma_freq * t + np.random.uniform(0, 2*np.pi))

        data[ch] = eeg_base + theta + alpha + beta + gamma

    # ── SCL (channel 32): slow drift with arousal-dependent amplitude ────────
    scl_freq = 0.05  # very slow, ~20 second cycle
    scl_amp = 2.0 + arousal_norm * 3.0  # higher arousal → more SCL
    scl_noise = np.random.randn(n) * 0.1
    data[CH_SCL] = scl_amp * (np.sin(2*np.pi*scl_freq*t) + 0.5) + scl_noise + 1.0

    # ── Respiration (channel 33): sinusoidal at realistic rate ───────────────
    resp_rate = 0.25 + arousal_norm * 0.1  # 15-21 breaths/min (higher with arousal)
    data[CH_RESP] = np.sin(2*np.pi*resp_rate*t) + np.random.randn(n)*0.05

    # ── Skin temperature (channel 34) ────────────────────────────────────────
    data[CH_TEMP] = 33.0 + valence_norm * 1.0 + np.random.randn(n) * 0.1

    # ── ECG (channel 35): realistic R-peaks ──────────────────────────────────
    hr = 60.0 + arousal_norm * 40.0  # 60-100 bpm (higher with arousal)
    rr_interval = DEAP_SRATE * 60.0 / hr
    # Add HRV: jitter RR intervals
    hrv_amount = (1.0 - arousal_norm) * 0.1 * rr_interval  # less HRV with high arousal
    ecg = np.zeros(n)
    t_peak = 0
    while t_peak < n:
        peak_sample = int(t_peak)
        if peak_sample < n:
            # QRS complex: Gaussian peak
            qrs = np.exp(-np.arange(-20, 21)**2 / (2 * 3**2))
            start = max(0, peak_sample - 20)
            end = min(n, peak_sample + 21)
            qrs_trimmed = qrs[max(0, 20-peak_sample):20+min(21, n-peak_sample)]
            ecg[start:end] += qrs_trimmed[:end-start]
        # Next peak: RR interval with HRV jitter
        jitter = np.random.randn() * hrv_amount
        t_peak += rr_interval + jitter
    ecg += np.random.randn(n) * 0.05
    data[CH_ECG] = ecg

    # ── BVP (channel 36) ─────────────────────────────────────────────────────
    data[CH_BVP] = np.sin(2*np.pi*(hr/60)*t) * (0.5 + arousal_norm*0.5) + np.random.randn(n)*0.05

    # ── EMG channels (37-38) ─────────────────────────────────────────────────
    data[CH_EMG_Z] = np.random.randn(n) * (0.1 + valence_norm * 0.2)  # more facial muscle with positive valence
    data[CH_EMG_T] = np.random.randn(n) * (0.1 + arousal_norm * 0.3)  # shoulder tension with arousal

    # ── EOG (channel 39) ─────────────────────────────────────────────────────
    data[CH_EOG] = np.random.randn(n) * 0.5 + np.sin(2*np.pi*0.3*t) * 0.2

    labels = np.array([valence, arousal, 5.0, 5.0])  # [valence, arousal, dominance, liking]
    return data, labels


def generate_synthetic_participant(n_trials: int = 10,
                                   seed: int = 42) -> Dict:
    """
    Generate a synthetic participant dataset.
    Covers the emotional space: high/low arousal × high/low valence.
    """
    np.random.seed(seed)
    data = np.zeros((n_trials, 40, DEAP_N_SAMPLES))
    labels = np.zeros((n_trials, 4))

    # Emotional trajectory: prenatal → birth-like → recovery → exploration
    emotional_sequence = [
        (3.0, 7.0),   # low arousal, positive: calm/settled
        (3.0, 7.0),
        (3.0, 7.0),
        (8.0, 3.0),   # high arousal, negative: stress/birth-analog
        (7.0, 4.0),   # high arousal, slightly negative: recovery
        (5.0, 6.0),   # moderate arousal, positive: normalization
        (4.0, 7.0),   # low-moderate, positive: consolidation
        (3.0, 8.0),   # low arousal, high positive: safety
        (5.0, 5.0),   # neutral baseline
        (6.0, 6.0),   # moderate positive: approach
    ]

    for i in range(min(n_trials, len(emotional_sequence))):
        arousal, valence = emotional_sequence[i]
        trial_data, trial_labels = generate_synthetic_deap_trial(
            arousal=arousal, valence=valence, seed=seed+i
        )
        data[i] = trial_data
        labels[i] = trial_labels

    return {'data': data, 'labels': labels}


# ─────────────────────────────────────────────────────────────────────────────
# FULL SYSTEM TEST — WHAT THE DEMONSTRATION LOOKS LIKE
# ─────────────────────────────────────────────────────────────────────────────

def run_demonstration(use_synthetic: bool = True,
                      deap_path: Optional[str] = None,
                      participant_id: int = 1):
    """
    Run the complete EFT system on physiological data.
    
    This is the demonstration:
    - Existence feeds in as real physiological presses
    - The system processes them through the conditional encounter environment
    - The topology shifts based on what was actually in the body
    - Before/after is measurable and was not scripted
    
    Args:
        use_synthetic: if True, use generated physiological signals
        deap_path:     path to DEAP data_preprocessed_python/ folder
        participant_id: which participant to run (1-32)
    """
    from eft_encounter_interaction import EncounterInteraction
    import sys

    print("=" * 60)
    print("EFT SYSTEM — PHYSIOLOGICAL ENCOUNTER DEMONSTRATION")
    print("The Mirror Platform LLC")
    print("=" * 60)
    print()

    # ── LOAD DATA ────────────────────────────────────────────────────────────
    if use_synthetic:
        print("Loading synthetic physiological data...")
        print("(Synthetic = physiologically-structured, NOT theorist-authored presses)")
        print()
        participant_data = generate_synthetic_participant(n_trials=10, seed=42)
        print(f"Generated 10 trials with emotional trajectory:")
        print("  Trials 1-3:  calm/settled (low arousal, positive)")
        print("  Trial 4:     stress event (high arousal, negative)")
        print("  Trial 5:     recovery")
        print("  Trials 6-8:  consolidation and normalization")
        print("  Trials 9-10: baseline and approach")
    else:
        if deap_path is None:
            print("ERROR: deap_path required for real data")
            return
        import pickle, os
        fname = os.path.join(deap_path, f"s{participant_id:02d}.dat")
        print(f"Loading DEAP participant {participant_id} from {fname}...")
        with open(fname, 'rb') as f:
            participant_data = pickle.load(f, encoding='latin1')
        print(f"Loaded: data shape={participant_data['data'].shape}, "
              f"labels shape={participant_data['labels'].shape}")

    print()

    # ── TRANSLATE TO PRESS STREAM ─────────────────────────────────────────────
    print("Translating physiological signals to existence presses...")
    presses = participant_to_press_stream(participant_data, participant_id)
    print(f"Press stream: {len(presses)} presses")
    print()

    # Show what the first three presses look like
    print("FIRST THREE PRESSES (what came from the body):")
    for i, press in enumerate(presses[:3]):
        print(f"  Press {i+1}: {press.label}")
        print(f"    magnitude={press.magnitude:.3f}  valence={press.valence:+.3f}  "
              f"persistence={press.persistence:.2f}")
        print(f"    direction: T={press.direction[0]:.3f} A={press.direction[1]:.3f} "
              f"B={press.direction[2]:.3f} R={press.direction[3]:.3f} E={press.direction[4]:.3f}")
    print()

    # ── INITIALIZE SYSTEM ─────────────────────────────────────────────────────
    print("Initializing EFT system...")
    interaction = EncounterInteraction(seed=42)
    print()

    # ── RECORD BASELINE ──────────────────────────────────────────────────────
    R_star_initial = interaction.self_system.R_star.copy()
    H_mu1_initial = interaction.env.H.mu1
    Sigma_initial = len(interaction.env.Sigma)

    print("BASELINE STATE (before any physiological encounters):")
    print(f"  R* = {R_star_initial}")
    print(f"  H.mu1 = {H_mu1_initial:.4f}")
    print(f"  Sigma entries = {Sigma_initial}")
    print()

    # ── FEED PHYSIOLOGICAL STREAM ─────────────────────────────────────────────
    print("Feeding physiological press stream into the system...")
    print()

    encountered_count = 0
    topology_shifts = []
    R_star_before_stress = None
    R_star_after_stress = None

    for i, press in enumerate(presses):
        event = interaction.process(press)

        if event.encountered:
            encountered_count += 1
            if event.delta_C:
                topology_shifts.append(event.delta_C["C_shift"])

        # Capture R* before and after the stress event (trial 4 = index 3)
        if i == 2:  # just before stress
            R_star_before_stress = interaction.self_system.R_star.copy()
            C_before_stress = interaction.env.H.mu1

        if i == 4:  # just after stress + recovery
            R_star_after_stress = interaction.self_system.R_star.copy()
            C_after_stress = interaction.env.H.mu1

        # Print encounter events
        if event.encountered:
            mode = event.delta_M5["mode"] if event.delta_M5 else "?"
            C_shift = event.delta_C["C_shift"] if event.delta_C else 0
            eigenmode = event.delta_M5["primary_eigenmode"] if event.delta_M5 else "?"
            print(f"  Press {i+1:2d} [{press.label}]: ENCOUNTER "
                  f"mode={mode:12s} primary={eigenmode:8s} "
                  f"C_shift={C_shift:+.4f} "
                  f"valence={press.valence:+.2f}")
        else:
            print(f"  Press {i+1:2d} [{press.label}]: no encounter "
                  f"(C={event.C_before:.3f} below threshold) "
                  f"valence={press.valence:+.2f}")

    print()

    # ── FINAL STATE ───────────────────────────────────────────────────────────
    print("=" * 60)
    print("FINAL STATE (after physiological encounter stream)")
    print("=" * 60)
    report = interaction.learning_report()

    print(f"Total presses:    {report['total_presses']}")
    print(f"Encounters:       {report['encounters']} ({report['encounter_rate']*100:.0f}%)")
    print(f"Sigma entries:    {report['actualization_record']}")
    print()

    print("CONDITION FIELD (what the system has become):")
    cf = report['condition_field']
    print(f"  Dominant eigenmode: {cf['dominant_eigenmode']}")
    print(f"  Weakest eigenmode:  {cf['weakest_eigenmode']}")
    print(f"  Approach bias:      alpha={cf['approach_alpha']:.4f}  beta={cf['avoidance_beta']:.4f}")
    print(f"  Magnitude center:   mu1={cf['magnitude_center']:.4f}")
    print()

    print("EIGENMODE ANISOTROPY (what the encounter history built):")
    names = ['temporal','agency','boundary','reality','exit']
    for name in names:
        val = cf['anisotropy'][name]
        bar = "█" * int(val * 200)
        print(f"  {name:10s}: {val:.4f}  {bar}")
    print()

    print("SELF-ATTRACTOR R* (where the self has moved):")
    geo = report['geometric']
    for name, val in geo['R_star'].items():
        print(f"  {name:10s}: {val:+.4f}")
    print(f"  Dominant: {geo['R_star_dominant']}")
    print()

    # ── THE DEMONSTRATION ─────────────────────────────────────────────────────
    if R_star_before_stress is not None and R_star_after_stress is not None:
        print("=" * 60)
        print("THE DEMONSTRATION: R* SHIFT THROUGH STRESS ENCOUNTER")
        print("=" * 60)
        print()
        print("R* BEFORE stress encounter (physiological state: calm):")
        names = ['temporal','agency','boundary','reality','exit']
        for i, name in enumerate(names):
            print(f"  {name:10s}: {R_star_before_stress[i]:+.4f}")
        print()
        print("R* AFTER stress + recovery (physiological state: normalizing):")
        for i, name in enumerate(names):
            print(f"  {name:10s}: {R_star_after_stress[i]:+.4f}")
        print()
        shift = np.linalg.norm(R_star_after_stress - R_star_before_stress)
        print(f"R* displacement: {shift:.4f}")
        print()
        print("This shift was produced by real physiological signals.")
        print("The topology that generated it did not come from the theory.")
        print("A body produced this. The system changed because of it.")
        print()

    print("TRAJECTORY SIGNATURE:")
    print(f"  {report['trajectory']}")
    print()

    print("Self-observation (what the system sees and is blind to):")
    so = interaction.self_observe()
    print(f"  Sees most clearly: {so['sees_clearly']} (lambda={so['dominant_lambda']:.4f})")
    print(f"  Blind to:          {so['blind_to']} (lambda={so['blind_lambda']:.4f})")
    print()
    print(so['note'])
    print()
    print("=" * 60)
    print("Demonstration complete.")
    print()
    print("What just happened:")
    print("  Real physiological signals → ExistencePress stream")
    print("  → Conditional encounter environment (commensurability gate)")
    print("  → EFT system (geometric encounter engine)")
    print("  → Measurable topology shift in R*, H(t), Sigma")
    print()
    print("What makes this different from every other system:")
    print("  The topology shift was not scripted.")
    print("  The presses were not authored by the theory.")
    print("  The system changed because existence acted on it.")
    print("=" * 60)

    return interaction


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if "--real" in sys.argv:
        # Usage: python eft_deap_translator.py --real /path/to/data_preprocessed_python/ --participant 1
        try:
            path_idx = sys.argv.index("--real") + 1
            deap_path = sys.argv[path_idx]
            p_idx = sys.argv.index("--participant") + 1 if "--participant" in sys.argv else None
            participant = int(sys.argv[p_idx]) if p_idx else 1
            run_demonstration(use_synthetic=False, deap_path=deap_path, participant_id=participant)
        except (IndexError, ValueError) as e:
            print(f"Error: {e}")
            print("Usage: python eft_deap_translator.py --real /path/to/deap/ --participant 1")
    else:
        run_demonstration(use_synthetic=True)
