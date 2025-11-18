"""
Anomaly Detection for Oxygen Sensor Data

Uses Isolation Forest to detect all types of anomalies through feature engineering.
The idea is simple: anomalies are isolated in high-dimensional feature space.

Quick Usage:
    # Load models
    from oxytrace.src.feature_engineering import OxygenFeatureEngineer
    from oxytrace.src.models.unified_anomaly_detector import UnifiedAnomalyDetector

    fe = OxygenFeatureEngineer.load('artifacts/anomaly_detector/feature_engineer.pkl')
    detector = UnifiedAnomalyDetector.load('artifacts/anomaly_detector/anomaly_detector.pkl')

    # Single prediction
    result = UnifiedAnomalyDetector.predict_single(
        oxygen_data={'time': '2024-01-01 10:00:00', 'Oxygen[%sat]': 88.5},
        feature_engineer=fe,
        detector=detector
    )
    # Returns: {'anomaly_score': 0.123, 'severity': 0, 'is_anomaly': False, 'label': 'normal'}
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class UnifiedAnomalyDetector:
    """Detects anomalies in oxygen sensor data using Isolation Forest."""

    def __init__(self, contamination=0.05, n_estimators=100, max_samples=256, random_state=42):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1,
        )

        self.is_fitted_ = False
        self.feature_names_ = None

    def fit(self, X, feature_names=None):
        """Train the model on normal data (learns what normal looks like)."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X_array = X.values
        else:
            self.feature_names_ = feature_names
            X_array = X

        # Clean data - remove NaN/Inf
        valid_mask = np.isfinite(X_array).all(axis=1)
        X_clean = X_array[valid_mask]

        # Scale features
        X_scaled = self.scaler.fit_transform(X_clean)

        # Train model
        self.model.fit(X_scaled)
        self.is_fitted_ = True

        return self

    def predict_anomaly_score(self, X):
        """Get anomaly score (0=normal, 1=anomaly) for each sample."""
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before prediction")

        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X

        valid_mask = np.isfinite(X_array).all(axis=1)

        # Scale features
        X_scaled = self.scaler.transform(X_array)

        # Get scores from Isolation Forest
        raw_scores = self.model.score_samples(X_scaled)

        # Convert to 0-1 range (higher = more anomalous)
        min_score = raw_scores.min()
        max_score = raw_scores.max()

        if max_score - min_score > 0:
            anomaly_scores = 1 - (raw_scores - min_score) / (max_score - min_score)
        else:
            anomaly_scores = np.zeros_like(raw_scores)

        anomaly_scores[~valid_mask] = 1.0
        return anomaly_scores

    def predict_anomaly_label(self, X, threshold=0.5):
        """Get binary labels (0=normal, 1=anomaly) using threshold."""
        scores = self.predict_anomaly_score(X)
        return (scores >= threshold).astype(int)

    def predict_with_severity(self, X):
        """
        Get scores and severity levels.
        Severity: 0=normal, 1=mild, 2=moderate, 3=severe
        """
        scores = self.predict_anomaly_score(X)

        severity = np.zeros_like(scores, dtype=int)
        severity[(scores >= 0.3) & (scores < 0.5)] = 1  # Mild
        severity[(scores >= 0.5) & (scores < 0.7)] = 2  # Moderate
        severity[scores >= 0.7] = 3  # Severe

        return scores, severity

    @classmethod
    def predict_single(cls, oxygen_data, feature_engineer, detector, threshold=0.25):
        """
        Predict anomaly for a single input or batch of oxygen readings.

        Args:
            oxygen_data: Can be:
                - DataFrame with 'time' and 'Oxygen[%sat]' columns
                - dict with 'time' and 'Oxygen[%sat]' keys
                - tuple/list of (time, oxygen_value) pairs
            feature_engineer: Trained OxygenFeatureEngineer instance
            detector: Trained UnifiedAnomalyDetector instance
            threshold: Anomaly threshold (default: 0.25)

        Returns:
            dict with keys:
                - 'anomaly_score': float (0-1, higher = more anomalous)
                - 'severity': int (0=normal, 1=mild, 2=moderate, 3=severe)
                - 'is_anomaly': bool (True if score > threshold)
                - 'label': str ('normal', 'mild', 'moderate', 'severe')
        """
        # Convert input to DataFrame
        if isinstance(oxygen_data, dict):
            df = pd.DataFrame([oxygen_data])
        elif isinstance(oxygen_data, (list, tuple)) and len(oxygen_data) == 2:
            df = pd.DataFrame({"time": [oxygen_data[0]], "Oxygen[%sat]": [oxygen_data[1]]})
        elif isinstance(oxygen_data, pd.DataFrame):
            df = oxygen_data.copy()
        else:
            raise ValueError("oxygen_data must be dict, (time, value) tuple, or DataFrame")

        # Ensure time is datetime
        if "time" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["time"]):
            df["time"] = pd.to_datetime(df["time"])

        # Engineer features
        features = feature_engineer.transform(df)

        # Predict
        scores, severity = detector.predict_with_severity(features)

        # Format results
        severity_labels = {0: "normal", 1: "mild", 2: "moderate", 3: "severe"}

        if len(scores) == 1:
            # Single prediction
            return {
                "anomaly_score": float(scores[0]),
                "severity": int(severity[0]),
                "is_anomaly": bool(scores[0] > threshold),
                "label": severity_labels[int(severity[0])],
            }
        else:
            # Batch prediction
            results = []
            for i in range(len(scores)):
                results.append(
                    {
                        "anomaly_score": float(scores[i]),
                        "severity": int(severity[i]),
                        "is_anomaly": bool(scores[i] > threshold),
                        "label": severity_labels[int(severity[i])],
                    }
                )
            return results

    def save(self, filepath):
        """Save model to disk."""
        model_data = {
            "scaler": self.scaler,
            "model": self.model,
            "feature_names": self.feature_names_,
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "random_state": self.random_state,
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, filepath)

    @classmethod
    def load(cls, filepath):
        """Load model from disk."""
        model_data = joblib.load(filepath)

        detector = cls(
            contamination=model_data["contamination"],
            n_estimators=model_data["n_estimators"],
            max_samples=model_data["max_samples"],
            random_state=model_data["random_state"],
        )

        detector.scaler = model_data["scaler"]
        detector.model = model_data["model"]
        detector.feature_names_ = model_data["feature_names"]
        detector.is_fitted_ = True

        return detector
