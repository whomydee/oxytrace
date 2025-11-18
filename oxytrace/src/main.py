"""
OxyTrace - Oxygen Sensor Anomaly Detection & Forecasting

Main application entry point demonstrating the complete workflow.
"""

import argparse
from pathlib import Path
import pandas as pd

from .feature_engineering import OxygenFeatureEngineer
from .models.unified_anomaly_detector import UnifiedAnomalyDetector
from .utils.custom_logger import LOGGER


def demo_workflow(data_percent=5.0):
    """
    Demonstrate the complete OxyTrace workflow.

    Args:
        data_percent: Percentage of data to use for demo
    """
    LOGGER.info("=" * 80)
    LOGGER.info("OXYTRACE DEMO - Complete Workflow")
    LOGGER.info("=" * 80)

    # Load data
    LOGGER.info("Loading data", percent=data_percent)
    df = pd.read_csv('oxytrace/dataset/dataset.csv')
    
    # Sample data
    sample_size = int(len(df) * data_percent / 100)
    df = df.head(sample_size)
    
    # Filter to oxygen readings only
    df = df[df['Oxygen[%sat]'].notna()].copy()
    
    LOGGER.info(
        "Data loaded",
        total_rows=len(df),
        oxygen_readings=df['Oxygen[%sat]'].count()
    )

    # Check if trained model exists
    model_path = Path("artifacts/anomaly_detector/anomaly_detector.pkl")
    feature_path = Path("artifacts/anomaly_detector/feature_engineer.pkl")

    if model_path.exists() and feature_path.exists():
        LOGGER.info("Loading pre-trained model")
        feature_engineer = OxygenFeatureEngineer.load(str(feature_path))
        detector = UnifiedAnomalyDetector.load(str(model_path))
        LOGGER.info("Artifacts relevant for prediction loaded")
    else:
        LOGGER.info("No pre-trained model found. Please run: python oxytrace/src/train.py")
        return

    # Engineer features
    LOGGER.info("Engineering features")
    features = feature_engineer.transform(df)
    
    # Run anomaly detection
    LOGGER.info("Running anomaly detection")
    scores, severity = detector.predict_with_severity(features)
    
    # Add results to dataframe
    df['anomaly_score'] = scores
    df['severity'] = severity
    
    # Show statistics
    anomaly_count = (scores > 0.25).sum()  # Using optimal threshold
    anomaly_rate = (scores > 0.25).mean() * 100
    
    LOGGER.info(
        "Anomaly detection complete",
        anomalies_detected=int(anomaly_count),
        anomaly_rate=f"{anomaly_rate:.2f}%"
    )
    
    # Severity breakdown
    severity_dist = {
        'normal': (severity == 0).sum(),
        'mild': (severity == 1).sum(),
        'moderate': (severity == 2).sum(),
        'severe': (severity == 3).sum()
    }
    LOGGER.info("Severity distribution", distribution=severity_dist)

    LOGGER.info("=" * 80)
    LOGGER.info("Demo complete!")
    LOGGER.info("=" * 80)


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description="OxyTrace - Oxygen Anomaly Detection & Forecasting")

    parser.add_argument("--demo", action="store_true", help="Run demo workflow")

    parser.add_argument(
        "--data-percent", type=float, default=5.0, help="Percentage of data to use for demo (default: 5%%)"
    )

    args = parser.parse_args()

    if args.demo:
        demo_workflow(args.data_percent)
    else:
        # Default: show help
        LOGGER.info("OxyTrace - Oxygen Sensor Anomaly Detection & Forecasting")
        LOGGER.info("")
        LOGGER.info("Usage:")
        LOGGER.info("  python -m oxytrace.src.main --demo              # Run demo workflow")
        LOGGER.info("  python oxytrace/src/train.py                    # Train models")
        LOGGER.info("")
        LOGGER.info("Or use Makefile commands:")
        LOGGER.info("  make train      # Train models")
        LOGGER.info("  make notebook   # Launch Jupyter")
        LOGGER.info("  make help       # Show all commands")


if __name__ == "__main__":
    main()
