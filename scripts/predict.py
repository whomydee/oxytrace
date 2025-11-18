"""
Inference script for oxygen anomaly detection

Loads trained model and makes predictions on new data.
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


def predict_anomalies(df, feature_engineer, detector, oxygen_col="Oxygen[%sat]"):
    """
    Predict anomalies for oxygen sensor data.

    Returns:
        DataFrame with original data plus anomaly_score and anomaly_severity columns
    """
    # Engineer features
    features = feature_engineer.transform(df, oxygen_col)

    # Get predictions
    scores, severity = detector.predict_with_severity(features)

    # Add to dataframe
    result = df.copy()
    result["anomaly_score"] = scores
    result["anomaly_severity"] = severity

    # Add human-readable severity labels
    severity_map = {0: "normal", 1: "mild", 2: "moderate", 3: "severe"}
    result["severity_label"] = result["anomaly_severity"].map(severity_map)

    return result


def main():
    print("=" * 60)
    print("OXYGEN ANOMALY DETECTION - INFERENCE")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    feature_engineer, detector = load_model()
    print("Model loaded successfully")

    # Load data
    print("\nLoading data...")
    df = pd.read_csv("oxytrace/dataset/dataset.csv")
    df["time"] = pd.to_datetime(df["time"], format="mixed")
    df = df[df["Oxygen[%sat]"].notna()].copy()
    df = df.sort_values("time").reset_index(drop=True)

    # Just predict on a sample for demo
    sample_df = df.head(10000)
    print(f"Predicting on {len(sample_df):,} samples...")

    # Make predictions
    results = predict_anomalies(sample_df, feature_engineer, detector)

    # Summary
    print("\n" + "=" * 60)
    print("PREDICTION SUMMARY")
    print("=" * 60)

    print(f"\nTotal samples: {len(results):,}")
    print(f"\nAnomaly Score Distribution:")
    print(f"  Mean: {results['anomaly_score'].mean():.3f}")
    print(f"  Median: {results['anomaly_score'].median():.3f}")
    print(f"  Max: {results['anomaly_score'].max():.3f}")

    print(f"\nSeverity Distribution:")
    severity_counts = results["severity_label"].value_counts()
    for severity in ["normal", "mild", "moderate", "severe"]:
        count = severity_counts.get(severity, 0)
        pct = (count / len(results)) * 100
        print(f"  {severity.capitalize():10s}: {count:6,} ({pct:5.1f}%)")

    # Show some anomalies
    anomalies = results[results["anomaly_score"] > 0.7].sort_values("anomaly_score", ascending=False)

    if len(anomalies) > 0:
        print(f"\nTop 5 Anomalies:")
        print(anomalies[["time", "Oxygen[%sat]", "anomaly_score", "severity_label"]].head().to_string(index=False))

    # Save results
    output_path = "analysis_data/predictions.csv"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
