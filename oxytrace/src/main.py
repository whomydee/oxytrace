"""
OxyTrace - Oxygen Sensor Anomaly Detection & Forecasting

Main application entry point demonstrating the complete workflow.
"""

import argparse
from pathlib import Path

from .utils.custom_logger import LOGGER
from .utils.dataset_util import DatasetUtil
from .anomaly_detector import AnomalyDetector


def demo_workflow(data_percent=5.0):
    """
    Demonstrate the complete OxyTrace workflow.
    
    Args:
        data_percent: Percentage of data to use for demo
    """
    LOGGER.info("=" * 80)
    LOGGER.info("OXYTRACE DEMO - Complete Workflow")
    LOGGER.info("=" * 80)
    
    # Load and preprocess data
    LOGGER.info("Loading and preprocessing data", percent=data_percent)
    df_raw = DatasetUtil.load_dataset(percent_of_data=data_percent)
    df_processed = DatasetUtil.preprocess_dataset(df_raw)
    
    LOGGER.info(
        "Data loaded",
        shape=df_processed.shape,
        columns=list(df_processed.columns),
        date_range=f"{df_processed['time'].min()} to {df_processed['time'].max()}"
    )
    
    # Check if trained model exists
    model_path = Path('artifacts/anomaly_detector/anomaly_detector.pkl')
    
    if model_path.exists():
        LOGGER.info("Loading pre-trained model")
        detector = AnomalyDetector()
        detector.load(str(model_path))
        LOGGER.info("✓ Model loaded")
    else:
        LOGGER.info("No pre-trained model found. Training on sample data...")
        detector = AnomalyDetector()
        
        # Use first 70% for training
        train_size = int(len(df_processed) * 0.7)
        train_df = df_processed.iloc[:train_size]
        
        LOGGER.info("Training detector", train_samples=len(train_df))
        detector.fit(train_df)
        LOGGER.info("✓ Detector trained")
    
    # Run anomaly detection on sample
    LOGGER.info("Running anomaly detection")
    sample_df = df_processed.head(1000)
    results_df = detector.detect(sample_df)
    
    anomaly_count = results_df['is_anomaly'].sum()
    anomaly_rate = results_df['is_anomaly'].mean() * 100
    
    LOGGER.info(
        "Anomaly detection complete",
        anomalies_detected=int(anomaly_count),
        anomaly_rate=f"{anomaly_rate:.2f}%"
    )
    
    if anomaly_count > 0:
        type_dist = results_df[results_df['is_anomaly']]['anomaly_type'].value_counts().to_dict()
        LOGGER.info("Anomaly types detected", distribution=type_dist)
                
        
    
    LOGGER.info("=" * 80)
    LOGGER.info("Demo complete!")
    LOGGER.info("=" * 80)
    


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description='OxyTrace - Oxygen Anomaly Detection & Forecasting')
    
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run demo workflow'
    )
    
    parser.add_argument(
        '--data-percent',
        type=float,
        default=5.0,
        help='Percentage of data to use for demo (default: 5%%)'
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

