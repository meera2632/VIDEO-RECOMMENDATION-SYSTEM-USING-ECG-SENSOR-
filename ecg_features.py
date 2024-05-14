"""ECG preprocessing and feature extraction used by training and inference."""
from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import butter, filtfilt, find_peaks, welch

FEATURE_NAMES = [
    "mean_hr", "std_hr", "mean_rr", "sdnn", "rmssd", "pnn50",
    "min_rr", "max_rr", "qrs_count", "signal_mean", "signal_std",
    "signal_energy", "signal_range", "lf_power", "hf_power", "lf_hf_ratio",
]


def clean_signal(signal: np.ndarray) -> np.ndarray:
    x = np.asarray(signal, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 10:
        raise ValueError("ECG window contains too few valid samples")
    return x


def bandpass_filter(signal: np.ndarray, sampling_rate: float,
                    low_hz: float = 0.5, high_hz: float = 40.0) -> np.ndarray:
    x = clean_signal(signal)
    nyquist = sampling_rate / 2.0
    high_hz = min(high_hz, nyquist * 0.9)
    if not 0 < low_hz < high_hz:
        raise ValueError("Invalid band-pass limits for the sampling rate")
    b, a = butter(3, [low_hz / nyquist, high_hz / nyquist], btype="band")
    padlen = min(len(x) - 1, 3 * max(len(a), len(b)))
    return filtfilt(b, a, x, padlen=padlen)


def detect_r_peaks(signal: np.ndarray, sampling_rate: float) -> np.ndarray:
    """Pan-Tompkins-style QRS detection using an energy envelope."""
    x = clean_signal(signal)
    filtered = bandpass_filter(x, sampling_rate)
    derivative = np.gradient(filtered)
    squared = derivative ** 2
    integration_samples = max(1, int(0.150 * sampling_rate))
    integrated = np.convolve(
        squared, np.ones(integration_samples) / integration_samples, mode="same"
    )
    distance = max(1, int(0.25 * sampling_rate))
    prominence = max(np.std(integrated) * 0.25, np.finfo(float).eps)
    candidates, _ = find_peaks(
        integrated, distance=distance, prominence=prominence
    )
    if candidates.size == 0:
        return candidates
    search = max(1, int(0.08 * sampling_rate))
    refined = []
    for peak in candidates:
        start = max(0, peak - search)
        end = min(len(filtered), peak + search + 1)
        refined.append(start + int(np.argmax(filtered[start:end])))
    return np.unique(refined)


def _hrv_features(r_peaks: np.ndarray, sampling_rate: float) -> dict[str, float]:
    if len(r_peaks) < 3:
        return {name: np.nan for name in FEATURE_NAMES[:9]}
    rr = np.diff(r_peaks) / sampling_rate
    rr = rr[(rr >= 0.30) & (rr <= 2.00)]
    if len(rr) < 2:
        return {name: np.nan for name in FEATURE_NAMES[:9]}
    hr = 60.0 / rr
    differences = np.diff(rr)
    return {
        "mean_hr": float(np.mean(hr)),
        "std_hr": float(np.std(hr)),
        "mean_rr": float(np.mean(rr)),
        "sdnn": float(np.std(rr, ddof=1)) if len(rr) > 1 else 0.0,
        "rmssd": float(np.sqrt(np.mean(differences ** 2))) if len(differences) else 0.0,
        "pnn50": float(np.mean(np.abs(differences) > 0.05)) if len(differences) else 0.0,
        "min_rr": float(np.min(rr)),
        "max_rr": float(np.max(rr)),
        "qrs_count": float(len(r_peaks)),
    }


def _frequency_features(r_peaks: np.ndarray, sampling_rate: float) -> dict[str, float]:
    if len(r_peaks) < 5:
        return {"lf_power": 0.0, "hf_power": 0.0, "lf_hf_ratio": 0.0}
    rr = np.diff(r_peaks) / sampling_rate
    times = r_peaks[1:] / sampling_rate
    if len(rr) < 4 or times[-1] <= times[0]:
        return {"lf_power": 0.0, "hf_power": 0.0, "lf_hf_ratio": 0.0}
    uniform_times = np.linspace(times[0], times[-1], max(16, len(times) * 4))
    uniform_rr = np.interp(uniform_times, times, rr)
    rr_rate = 1.0 / np.median(np.diff(uniform_times))
    frequencies, power = welch(
        uniform_rr - np.mean(uniform_rr),
        fs=rr_rate,
        nperseg=min(256, len(uniform_rr)),
    )
    lf_mask = (frequencies >= 0.04) & (frequencies < 0.15)
    hf_mask = (frequencies >= 0.15) & (frequencies <= 0.40)
    lf = float(trapezoid(power[lf_mask], frequencies[lf_mask])) if lf_mask.any() else 0.0
    hf = float(trapezoid(power[hf_mask], frequencies[hf_mask])) if hf_mask.any() else 0.0
    return {"lf_power": lf, "hf_power": hf, "lf_hf_ratio": lf / hf if hf > 0 else 0.0}


def extract_features(signal: np.ndarray, sampling_rate: float) -> dict[str, float]:
    x = clean_signal(signal)
    filtered = bandpass_filter(x, sampling_rate)
    r_peaks = detect_r_peaks(x, sampling_rate)
    features = {
        "signal_mean": float(np.mean(filtered)),
        "signal_std": float(np.std(filtered)),
        "signal_energy": float(np.mean(filtered ** 2)),
        "signal_range": float(np.ptp(filtered)),
    }
    features.update(_hrv_features(r_peaks, sampling_rate))
    features.update(_frequency_features(r_peaks, sampling_rate))
    return {name: float(features.get(name, np.nan)) for name in FEATURE_NAMES}


def window_signal(signal: np.ndarray, sampling_rate: float,
                  window_seconds: float = 10.0, step_seconds: float = 5.0):
    x = np.asarray(signal, dtype=float)
    window = int(window_seconds * sampling_rate)
    step = int(step_seconds * sampling_rate)
    if window <= 0 or step <= 0 or len(x) < window:
        return
    for start in range(0, len(x) - window + 1, step):
        yield x[start:start + window]
