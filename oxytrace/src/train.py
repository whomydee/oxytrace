"""
Training script for oxygen anomaly detection model
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from feature_engineering import OxygenFeatureEngineer
from models.unified_anomaly_detector import UnifiedAnomalyDetector


def load_data(filepath):
    """Load and prepare oxygen sensor data."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Convert time column
    df['time'] = pd.to_datetime(df['time'], format='mixed')
    
    # Filter to only oxygen readings
    df = df[df['Oxygen[%sat]'].notna()].copy()
    df = df.sort_values('time').reset_index(drop=True)
    
    print(f"Loaded {len(df):,} oxygen readings")
    return df


def train_anomaly_detector(df, output_dir='artifacts/anomaly_detector'):
    """Train and save anomaly detection model."""
    print("\n" + "="*60)
    print("TRAINING ANOMALY DETECTION MODEL")
    print("="*60)
    
    # Step 1: Feature engineering
    print("Step 1: Engineering features...")
    feature_engineer = OxygenFeatureEngineer(
        short_window=60,
        long_window=360,
        fault_window=30
    )
    
    features = feature_engineer.fit_transform(df)
    print(f"Created {features.shape[1]} features from {len(features):,} samples")
    
    # Step 2: Train anomaly detector
    print("Step 2: Training Isolation Forest...")
    detector = UnifiedAnomalyDetector(
        contamination=0.05,  # Expect ~5% anomalies
        n_estimators=100,
        max_samples=256,
        random_state=42
    )
    
    detector.fit(features)
    print("Model training complete")
    
    # Step 3: Save model components
    print("Step 3: Saving model...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save feature engineer
    feature_engineer.save(output_path / 'feature_engineer.pkl')
    
    # Save detector
    detector.save(output_path / 'anomaly_detector.pkl')
    
    print(f"Model saved to {output_path}")
    
    # Step 4: Quick validation
    print("Step 4: Running quick validation...")
    scores = detector.predict_anomaly_score(features)
    
    print(f"Anomaly score statistics:")
    print(f"  Mean: {scores.mean():.3f}")
    print(f"  Median: {np.median(scores):.3f}")
    print(f"  Std: {scores.std():.3f}")
    print(f"  Max: {scores.max():.3f}")
    print(f"  Samples with score > 0.5: {(scores > 0.5).sum():,} ({(scores > 0.5).mean()*100:.1f}%)")
    print(f"  Samples with score > 0.7: {(scores > 0.7).sum():,} ({(scores > 0.7).mean()*100:.1f}%)")
    
    return feature_engineer, detector


def main():
    # Load data
    data_path = 'oxytrace/dataset/dataset.csv'
    df = load_data(data_path)
    
    # Train model
    feature_engineer, detector = train_anomaly_detector(df)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print("Model artifacts saved in: artifacts/anomaly_detector/")
    print("  - feature_engineer.pkl")
    print("  - anomaly_detector.pkl")


if __name__ == '__main__':
    main()
