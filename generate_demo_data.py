"""Generate synthetic ECG-like CSV files for testing the pipeline.

These signals are only for software testing. They are not medically valid
ECG recordings and must not be used to claim model accuracy.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EMOTIONS = {
    "happy": {"heart_rate": 88, "noise": 0.035, "amplitude": 1.00},
    "relaxed": {"heart_rate": 68, "noise": 0.020, "amplitude": 0.85},
    "sad": {"heart_rate": 62, "noise": 0.030, "amplitude": 0.75},
    "angry": {"heart_rate": 108, "noise": 0.060, "amplitude": 1.10},
}


def synthetic_ecg(duration_seconds: float, sampling_rate: int,
                   heart_rate: float, noise_level: float,
                   amplitude: float, rng: np.random.Generator) -> np.ndarray:
    """Create a simple ECG-like waveform with visible QRS pulses."""
    count = int(duration_seconds * sampling_rate)
    time = np.arange(count) / sampling_rate
    rr_seconds = 60.0 / heart_rate
    beat_times = np.arange(0.4, duration_seconds, rr_seconds)
    signal = 0.03 * np.sin(2 * np.pi * 0.25 * time)

    for beat in beat_times:
        signal += 0.10 * np.exp(-0.5 * ((time - (beat - 0.20)) / 0.045) ** 2)
        signal -= 0.08 * np.exp(-0.5 * ((time - (beat - 0.035)) / 0.012) ** 2)
        signal += amplitude * np.exp(-0.5 * ((time - beat) / 0.018) ** 2)
        signal -= 0.15 * np.exp(-0.5 * ((time - (beat + 0.035)) / 0.014) ** 2)
        signal += 0.25 * amplitude * np.exp(-0.5 * ((time - (beat + 0.25)) / 0.080) ** 2)

    signal += rng.normal(0, noise_level, size=count)
    return signal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--sampling-rate", type=int, default=250)
    parser.add_argument("--subjects", type=int, default=3)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    for subject_number in range(1, args.subjects + 1):
        for emotion, settings in EMOTIONS.items():
            subject_id = f"subject_{subject_number:02d}"
            session_id = f"{subject_id}_{emotion}_session_01"
            signal = synthetic_ecg(
                args.duration,
                args.sampling_rate,
                settings["heart_rate"] + rng.normal(0, 2),
                settings["noise"],
                settings["amplitude"],
                rng,
            )
            frame = pd.DataFrame({
                "timestamp": np.arange(len(signal)) / args.sampling_rate,
                "sensor": signal,
                "emotion": emotion,
                "subject_id": subject_id,
                "session_id": session_id,
            })
            frame.to_csv(output_dir / f"{subject_id}_{emotion}.csv", index=False)

    print(f"Generated {args.subjects * len(EMOTIONS)} synthetic CSV files in {output_dir}")


if __name__ == "__main__":
    main()
