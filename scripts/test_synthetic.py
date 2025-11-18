"""
Test the model on synthetic anomaly data
"""

import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from oxytrace.src.models.unified_anomaly_detector import UnifiedAnomalyDetector


def load_model(model_dir="artifacts/anomaly_detector"):
    """Load trained model components."""
    model_path = Path(model_dir)

    feature_engineer = joblib.load(model_path / "feature_engineer.pkl")
    detector = UnifiedAnomalyDetector.load(model_path / "anomaly_detector.pkl")

    return feature_engineer, detector


def test_on_synthetic_data():
    print("=" * 60)
    print("TESTING ON SYNTHETIC DATA")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    feature_engineer, detector = load_model()

    # Load synthetic data
    print("Loading synthetic data...")
    df = pd.read_csv("outputs/synthetic_data/synthetic_anomalies.csv")
    df["time"] = pd.to_datetime(df["time"])

    # Engineer features
    print("Engineering features...")
    features = feature_engineer.transform(df)

    # Predict
    print("Running predictions...")
    scores, severity = detector.predict_with_severity(features)

    df["anomaly_score"] = scores
    df["anomaly_severity"] = severity

    # Evaluate performance by anomaly type
    print("\n" + "=" * 60)
    print("RESULTS BY ANOMALY TYPE")
    print("=" * 60)

    for anomaly_type in ["normal", "point_anomaly", "collective_anomaly", "stuck_sensor", "noisy_sensor"]:
        subset = df[df["anomaly_type"] == anomaly_type]
        if len(subset) == 0:
            continue

        print(f"\n{anomaly_type.upper()}:")
        print(f"  Count: {len(subset):,}")
        print(f"  Mean score: {subset['anomaly_score'].mean():.3f}")
        print(f"  Median score: {subset['anomaly_score'].median():.3f}")
        print(f"  Max score: {subset['anomaly_score'].max():.3f}")
        print(f"  % with score > 0.5: {(subset['anomaly_score'] > 0.5).mean() * 100:.1f}%")
        print(f"  % with score > 0.7: {(subset['anomaly_score'] > 0.7).mean() * 100:.1f}%")

    # Classification performance
    print("\n" + "=" * 60)
    print("CLASSIFICATION PERFORMANCE (threshold=0.5)")
    print("=" * 60)

    df["is_anomaly_true"] = df["anomaly_type"] != "normal"
    df["is_anomaly_pred"] = df["anomaly_score"] > 0.5

    tp = ((df["is_anomaly_true"]) & (df["is_anomaly_pred"])).sum()
    fp = ((~df["is_anomaly_true"]) & (df["is_anomaly_pred"])).sum()
    tn = ((~df["is_anomaly_true"]) & (~df["is_anomaly_pred"])).sum()
    fn = ((df["is_anomaly_true"]) & (~df["is_anomaly_pred"])).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(df)

    print(f"\nTrue Positives (detected anomalies): {tp}")
    print(f"False Positives (false alarms): {fp}")
    print(f"True Negatives (correct normals): {tn}")
    print(f"False Negatives (missed anomalies): {fn}")
    print(f"\nPrecision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")
    print(f"Accuracy: {accuracy:.3f}")

    # Save results
    output_path = "outputs/synthetic_test_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    test_on_synthetic_data()
