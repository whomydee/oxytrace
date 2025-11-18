"""
Training Pipeline for Anomaly Detection Model

Trains model on clean oxygen sensor data and saves artifacts.
"""

import argparse
from pathlib import Path
from datetime import datetime
import json

import pandas as pd

from oxytrace.src.utils.custom_logger import LOGGER
from oxytrace.src.utils.dataset_util import DatasetUtil
from oxytrace.src.anomaly_detector import AnomalyDetector


def create_directories():
    """Create necessary directories for models and outputs."""
    directories = [
        'artifacts',
        'artifacts/anomaly_detector',
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    LOGGER.info("Created output directories")


def train_anomaly_detector(
    train_df: pd.DataFrame,
    model_dir: str = 'artifacts/anomaly_detector',
) -> AnomalyDetector:
    """
    Train anomaly detection model.
    
    Args:
        train_df: Training dataframe (clean data only)
        model_dir: Directory to save model
        
    Returns:
        Trained AnomalyDetector
    """
    LOGGER.info("=" * 80)
    LOGGER.info("TRAINING ANOMALY DETECTOR")
    LOGGER.info("=" * 80)
    LOGGER.info("Training data rows", rows=len(train_df))
    
    # Initialize detector
    detector = AnomalyDetector(z_threshold=3.0)
    
    # Train with optimized epochs
    LOGGER.info("Starting training (this may take several minutes)...")
    detector.fit(train_df)
    
    # Save model
    model_path = Path(model_dir) / 'anomaly_detector.pkl'
    detector.save(str(model_path))
    
    LOGGER.info("✓ Anomaly detector saved", path=str(model_path))
    LOGGER.info("=" * 80)
    
    return detector


def main(args):
    """Main training pipeline."""
    LOGGER.info("=" * 80)
    LOGGER.info("Starting training pipeline")
    LOGGER.info("=" * 80)
    
    # Create directories
    create_directories()
    
    # Load data
    LOGGER.info("Loading dataset", percent=args.data_percent)
    df_raw = DatasetUtil.load_dataset(percent_of_data=args.data_percent)
    
    # Preprocess
    LOGGER.info("Preprocessing dataset")
    df_processed = DatasetUtil.preprocess_dataset(df_raw)
    
    # Split data chronologically
    LOGGER.info("Splitting data into train/val/test")
    train_df, val_df, test_df = DatasetUtil.split_train_test_val(
        df_processed,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    # Train anomaly detector
    if args.train_anomaly_detector:
        LOGGER.info("Training anomaly detection model")
        detector = train_anomaly_detector(
            train_df,
        )
        
        # Quick validation
        LOGGER.info("Running validation on clean data")
        val_results = detector.detect(val_df.head(1000))
        val_anomaly_rate = val_results['is_anomaly'].mean() * 100
        LOGGER.info("Validation anomaly rate (should be low)", rate=f"{val_anomaly_rate:.2f}%")
    
    # Save training metadata
    training_info = {
        'training_date': datetime.now().isoformat(),
        'data_percent_used': args.data_percent,
        'train_records': len(train_df),
        'val_records': len(val_df),
        'test_records': len(test_df),
        'models_trained': {
            'anomaly_detector': args.train_anomaly_detector,
        }
    }
    
    with open('artifacts/training_info.json', 'w') as f:
        json.dump(training_info, f, indent=2)
    
    LOGGER.info("=" * 80)
    LOGGER.info("Training pipeline completed successfully")
    LOGGER.info("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train anomaly detection model')
    
    parser.add_argument(
        '--data-percent',
        type=float,
        default=10.0,
        help='Percentage of data to use for training (default: 10%%)'
    )
    
    parser.add_argument(
        '--train-anomaly-detector',
        action='store_true',
        default=True,
        help='Train anomaly detection model'
    )
    
    args = parser.parse_args()
    main(args)
