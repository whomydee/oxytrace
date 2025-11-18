"""
Anomaly Detection for Oxygen Sensor Data

Uses Isolation Forest to detect all types of anomalies through feature engineering.
The idea is simple: anomalies are isolated in high-dimensional feature space.
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
