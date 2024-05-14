# ECG Video Recommendation System

This project combines an ESP32, AD8232 ECG sensor, ECG signal processing, supervised machine learning, KNN video recommendation, and MySQL.

## Architecture

    AD8232 -> ESP32 -> Ubidots or CSV
    CSV -> filtering -> R-peak detection -> HRV features
    HRV features -> SVM or Random Forest -> predicted emotion
    Predicted emotion -> valence/arousal target -> KNN
    KNN -> MySQL video URLs -> browser playback

The supported emotion classes are happy, relaxed, sad, and angry.

## Project structure

    ecg_video_recommender/
    ├── esp32_ecg.ino
    ├── ecg_features.py
    ├── train_model.py
    ├── predict_and_recommend.py
    ├── recommend_videos.py
    ├── generate_demo_data.py
    ├── requirements.txt
    ├── README.md
    ├── data/
    │   ├── subject_01_happy.csv
    │   └── ...
    └── models/
        ├── emotion_model.joblib
        ├── extracted_training_features.csv
        └── training_metadata.json

## File responsibilities

| File | Purpose |
|---|---|
| esp32_ecg.ino | Reads AD8232 data and publishes raw ECG values through MQTT |
| ecg_features.py | Filters ECG, detects R-peaks, and extracts HRV features |
| train_model.py | Trains and evaluates SVM or Random Forest |
| predict_and_recommend.py | Predicts emotion and starts recommendations |
| recommend_videos.py | Fits KNN on MySQL video metadata |
| generate_demo_data.py | Creates synthetic ECG-like files for software testing |

## Windows installation

Open PowerShell in the project folder:

    python -m venv .venv
    .\\.venv\\Scripts\\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

If activation is blocked:

    Set-ExecutionPolicy -Scope Process Bypass
    .\\.venv\\Scripts\\Activate.ps1

## Training-data format

Each CSV contains one ECG sample per row. Required columns are sensor and emotion.

    timestamp,sensor,emotion,subject_id,session_id
    0.000,1842,happy,subject_01,session_01
    0.004,1850,happy,subject_01,session_01
    0.008,1836,happy,subject_01,session_01

The emotion must be happy, relaxed, sad, or angry. Timestamp, subject_id, and session_id are strongly recommended. subject_id helps prevent participant-level data leakage during evaluation.

## Generate test data

    python generate_demo_data.py --output-dir data --duration 30 --sampling-rate 250 --subjects 3

This creates 12 files: 3 subjects multiplied by 4 emotions. The synthetic ECG is only for checking that the pipeline runs. It is not medically valid and must not support a real accuracy claim.

## Train the emotion model

    python train_model.py --data-dir data --sampling-rate 250 --model svm

Or use Random Forest:

    python train_model.py --data-dir data --sampling-rate 250 --model random_forest

Training performs filtering, Pan-Tompkins-style R-peak detection, HRV feature extraction, scaling, supervised ML training, and held-out evaluation.

The command creates:

    models/emotion_model.joblib
    models/extracted_training_features.csv
    models/training_metadata.json

## Predict and recommend videos

    python predict_and_recommend.py --csv data/live_ecg.csv --model models/emotion_model.joblib --open-videos

Omit --open-videos if you only want the URLs printed.

## MySQL video catalog

Create the table:

    CREATE TABLE videos (
        id INT AUTO_INCREMENT PRIMARY KEY,
        emotion VARCHAR(32) NOT NULL,
        video_url TEXT NOT NULL,
        valence FLOAT NOT NULL,
        arousal FLOAT NOT NULL
    );

KNN uses valence and arousal values from 0 to 1:

| Emotion | Valence | Arousal |
|---|---:|---:|
| Happy | 0.8 | 0.8 |
| Relaxed | 0.8 | 0.2 |
| Sad | 0.2 | 0.2 |
| Angry | 0.2 | 0.8 |

The classifier predicts an emotion, the code converts it into a valence/arousal target, and KNN returns the closest videos in the MySQL catalog.

## Database credentials

Set credentials in PowerShell instead of hardcoding them:

    $env:MYSQL_HOST = "localhost"
    $env:MYSQL_USER = "root"
    $env:MYSQL_PASSWORD = "your_password"
    $env:MYSQL_DATABASE = "beproject"

## ESP32 and AD8232

The ESP32 code reads the AD8232 output on GPIO 34, samples at approximately 250 Hz, checks lead-off pins, and publishes raw ADC values to Ubidots through MQTT. The Python pipeline currently expects Ubidots data to be exported or saved as CSV; automatic Ubidots downloading is not included.

## Validation and limitations

- Use DREAMER or properly labeled real ECG recordings for meaningful evaluation.
- Keep the sampling rate and preprocessing identical during training and prediction.
- Split by participant rather than randomly splitting rows.
- Report accuracy, precision, recall, macro F1-score, and a confusion matrix.
- Do not interpret the output as a medical diagnosis.

ECG features can be affected by movement, electrode placement, noise, and individual physiology. Emotion predictions are experimental estimates.
