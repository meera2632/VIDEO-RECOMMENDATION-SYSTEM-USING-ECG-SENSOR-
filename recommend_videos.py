"""KNN-based video recommendation backed by MySQL metadata."""
from __future__ import annotations

import os
import mysql.connector
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


EMOTION_TARGETS = {
    "happy": (0.8, 0.8),
    "relaxed": (0.8, 0.2),
    "sad": (0.2, 0.2),
    "angry": (0.2, 0.8),
}


def fetch_knn_video_urls(emotion: str, limit: int = 5):
    """Fit KNN on video valence/arousal metadata and return nearest videos."""
    if emotion not in EMOTION_TARGETS:
        raise ValueError(f"Unknown emotion: {emotion}")
    connection = mysql.connector.connect(
        host=os.getenv("MYSQL_HOST", "localhost"),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", ""),
        database=os.getenv("MYSQL_DATABASE", "beproject"),
    )
    try:
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT id, video_url, emotion, valence, arousal
            FROM videos
            WHERE valence IS NOT NULL AND arousal IS NOT NULL
            """
        )
        rows = cursor.fetchall()
        if not rows:
            return []

        video_features = np.asarray([[float(row[3]), float(row[4])] for row in rows])
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(video_features)
        knn = NearestNeighbors(
            n_neighbors=min(limit, len(rows)),
            metric="euclidean",
        )
        knn.fit(scaled_features)

        target = scaler.transform([EMOTION_TARGETS[emotion]])
        _, indices = knn.kneighbors(target)
        return [rows[index][1] for index in indices[0]]
    finally:
        connection.close()
