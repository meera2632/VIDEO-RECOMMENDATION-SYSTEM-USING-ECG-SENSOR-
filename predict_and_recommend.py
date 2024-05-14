"""Predict emotion from an ECG CSV and retrieve matching videos."""
from __future__ import annotations

import argparse
import webbrowser

import joblib
import pandas as pd

from ecg_features import extract_features, window_signal
from recommend_videos import fetch_knn_video_urls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--model", default="models/emotion_model.joblib")
    parser.add_argument("--sensor-column", default="sensor")
    parser.add_argument("--open-videos", action="store_true")
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    feature_names = bundle["feature_names"]
    df = pd.read_csv(args.csv)
    signal = pd.to_numeric(df[args.sensor_column], errors="coerce").dropna().to_numpy()
    sampling_rate = float(bundle["sampling_rate"])
    window_seconds = float(bundle["window_seconds"])
    pipeline = bundle["pipeline"]
    predictions = []

    for window in window_signal(signal, sampling_rate, window_seconds, window_seconds):
        feature_dict = extract_features(window, sampling_rate)
        X = pd.DataFrame([[feature_dict[name] for name in feature_names]], columns=feature_names)
        emotion = str(pipeline.predict(X)[0])
        confidence = float(max(pipeline.predict_proba(X)[0]))
        predictions.append((emotion, confidence))

    if not predictions:
        raise ValueError("The ECG file is shorter than one prediction window")

    emotion = max(
        set(item[0] for item in predictions),
        key=lambda value: sum(item[0] == value for item in predictions),
    )
    confidence_values = [value for label, value in predictions if label == emotion]
    print(f"Predicted emotion: {emotion}")
    print(f"Average confidence: {sum(confidence_values) / len(confidence_values):.2%}")

    try:
        urls = fetch_knn_video_urls(emotion)
    except Exception as exc:
        print(f"Database lookup failed: {exc}")
        return
    if not urls:
        print(f"No videos found for emotion: {emotion}")
        return
    for url in urls:
        print(f"Recommended video: {url}")
        if args.open_videos:
            webbrowser.open(url)


if __name__ == "__main__":
    main()
