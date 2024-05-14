"""Train and evaluate an ECG emotion classifier."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ecg_features import FEATURE_NAMES, extract_features, window_signal

VALID_LABELS = {"happy", "relaxed", "sad", "angry"}


def load_windows(data_dir: Path, sampling_rate: float,
                 window_seconds: float, step_seconds: float) -> pd.DataFrame:
    rows = []
    for path in sorted(data_dir.glob("*.csv")):
        df = pd.read_csv(path)
        if "sensor" not in df.columns:
            raise ValueError(f"{path} must contain a 'sensor' column")
        if "emotion" in df.columns:
            labels = df["emotion"].dropna().astype(str).str.lower().str.strip()
            if labels.empty:
                raise ValueError(f"{path} has no emotion labels")
            label = labels.mode().iloc[0]
        else:
            label = path.stem.lower().strip()
        if label not in VALID_LABELS:
            raise ValueError(f"Unsupported label '{label}' in {path}; use {sorted(VALID_LABELS)}")
        signal = pd.to_numeric(df["sensor"], errors="coerce").dropna().to_numpy()
        group = str(df["subject_id"].iloc[0]) if "subject_id" in df.columns else path.stem
        if "session_id" in df.columns:
            group = f"{group}_{df['session_id'].iloc[0]}"
        for index, window in enumerate(
            window_signal(signal, sampling_rate, window_seconds, step_seconds)
        ):
            features = extract_features(window, sampling_rate)
            features.update({"emotion": label, "group": group, "source": path.name, "window": index})
            rows.append(features)
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No usable ECG windows were created")
    return result.replace([np.inf, -np.inf], np.nan)


def build_pipeline(model_name: str) -> Pipeline:
    classifier = (
        SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)
        if model_name == "svm"
        else RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
        )
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", classifier),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-out", default="models/emotion_model.joblib")
    parser.add_argument("--sampling-rate", type=float, default=250.0)
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--step-seconds", type=float, default=5.0)
    parser.add_argument("--model", choices=["svm", "random_forest"], default="svm")
    args = parser.parse_args()

    table = load_windows(
        Path(args.data_dir), args.sampling_rate,
        args.window_seconds, args.step_seconds
    )
    X = table[FEATURE_NAMES]
    y = table["emotion"]
    groups = table["group"]

    if groups.nunique() >= 2:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    else:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y))

    pipeline = build_pipeline(args.model)
    pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
    predictions = pipeline.predict(X.iloc[test_idx])
    print(classification_report(y.iloc[test_idx], predictions, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y.iloc[test_idx], predictions, labels=sorted(VALID_LABELS)))

    try:
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="f1_macro")
        print(f"5-fold macro F1: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
    except ValueError as exc:
        print(f"Cross-validation skipped: {exc}")

    output = Path(args.model_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    test_f1 = float(f1_score(y.iloc[test_idx], predictions, average="macro"))
    joblib.dump({
        "pipeline": pipeline,
        "feature_names": FEATURE_NAMES,
        "sampling_rate": args.sampling_rate,
        "window_seconds": args.window_seconds,
        "step_seconds": args.step_seconds,
        "labels": sorted(VALID_LABELS),
        "test_macro_f1": test_f1,
    }, output)
    table.to_csv(output.parent / "extracted_training_features.csv", index=False)
    (output.parent / "training_metadata.json").write_text(
        json.dumps({"rows": len(table), "model": args.model, "test_macro_f1": test_f1}, indent=2)
    )
    print(f"Saved model to {output}")


if __name__ == "__main__":
    main()

