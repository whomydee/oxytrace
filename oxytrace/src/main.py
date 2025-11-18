"""
OxyTrace - Oxygen Sensor Anomaly Detection & Forecasting

Main application entry point demonstrating the complete workflow.
"""

import argparse
from pathlib import Path

import pandas as pd

from .feature_engineering import OxygenFeatureEngineer
from .models.unified_anomaly_detector import UnifiedAnomalyDetector
from .models.simple_forecaster import SimpleOxygenForecaster
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
    df = pd.read_csv("oxytrace/dataset/dataset.csv")

    # Sample data
    sample_size = int(len(df) * data_percent / 100)
    df = df.head(sample_size)

    # Filter to oxygen readings only
    df = df[df["Oxygen[%sat]"].notna()].copy()

    LOGGER.info("Data loaded", total_rows=len(df), oxygen_readings=df["Oxygen[%sat]"].count())

    # Check if trained model exists
    model_path = Path("artifacts/anomaly_detector/anomaly_detector.pkl")
    feature_path = Path("artifacts/anomaly_detector/feature_engineer.pkl")

    if model_path.exists() and feature_path.exists():
        LOGGER.info("Loading pre-trained model")
        feature_engineer = OxygenFeatureEngineer.load(str(feature_path))
        detector = UnifiedAnomalyDetector.load(str(model_path))
        LOGGER.info("Artifacts relevant for prediction loaded")
    else:
        LOGGER.info("No pre-trained model found. Please run: python oxytrace/src/train_detector.py")
        return

    # Engineer features
    LOGGER.info("Engineering features")
    features = feature_engineer.transform(df)

    # Run anomaly detection
    LOGGER.info("Running anomaly detection")
    scores, severity = detector.predict_with_severity(features)

    # Add results to dataframe
    df["anomaly_score"] = scores
    df["severity"] = severity

    # Show statistics
    anomaly_count = (scores > 0.25).sum()  # Using optimal threshold
    anomaly_rate = (scores > 0.25).mean() * 100

    LOGGER.info(
        "Anomaly detection complete", anomalies_detected=int(anomaly_count), anomaly_rate=f"{anomaly_rate:.2f}%"
    )

    # Severity breakdown
    severity_dist = {
        "normal": (severity == 0).sum(),
        "mild": (severity == 1).sum(),
        "moderate": (severity == 2).sum(),
        "severe": (severity == 3).sum(),
    }
    LOGGER.info("Severity distribution", distribution=severity_dist)

    LOGGER.info("=" * 80)
    LOGGER.info("Demo complete!")
    LOGGER.info("=" * 80)


def train_and_forecast(horizon_days=7, train_first=False):
    """
    Generate forecasts (optionally training first).
    
    Args:
        horizon_days: Number of days to forecast ahead
        train_first: If True, train new model even if one exists
    """
    LOGGER.info("=" * 80)
    LOGGER.info("FORECASTING WORKFLOW")
    LOGGER.info("=" * 80)
    
    forecaster_path = Path("artifacts/forecaster/oxygen_forecaster.pkl")
    
    # Train if requested or no model exists
    if train_first or not forecaster_path.exists():
        LOGGER.info("Loading data")
        df = pd.read_csv("oxytrace/dataset/dataset.csv")
        df['time'] = pd.to_datetime(df['time'], format='mixed')
        df = df[df['Oxygen[%sat]'].notna()].copy()
        df = df.sort_values('time').reset_index(drop=True)
        
        LOGGER.info("Data loaded", total_samples=len(df))
        LOGGER.info("Training forecaster")
        
        forecaster = SimpleOxygenForecaster()
        forecaster.fit(df)
        
        forecaster_path.parent.mkdir(parents=True, exist_ok=True)
        forecaster.save(str(forecaster_path))
        LOGGER.info("Forecaster trained and saved", path=str(forecaster_path))
    else:
        LOGGER.info("Loading pre-trained forecaster")
        forecaster = SimpleOxygenForecaster.load(str(forecaster_path))
        LOGGER.info("Forecaster loaded")
    
    # Make predictions
    horizon = horizon_days * 1440  # Convert days to minutes
    LOGGER.info("Generating forecast", horizon_days=horizon_days, horizon_minutes=horizon)
    
    forecast_df = forecaster.predict(horizon=horizon, return_uncertainty=True)
    
    # Show forecast summary
    mean_pred = forecast_df['predicted_oxygen'].mean()
    std_pred = forecast_df['predicted_oxygen'].std()
    min_pred = forecast_df['predicted_oxygen'].min()
    max_pred = forecast_df['predicted_oxygen'].max()
    
    LOGGER.info(
        "Forecast complete",
        mean=f"{mean_pred:.2f}%",
        std=f"{std_pred:.2f}%",
        range=f"{min_pred:.2f}% to {max_pred:.2f}%"
    )
    
    # Save forecast to CSV
    output_path = Path("artifacts/forecaster/latest_forecast.csv")
    forecast_df.to_csv(output_path, index=False)
    LOGGER.info("Forecast saved", path=str(output_path))
    
    LOGGER.info("=" * 80)
    LOGGER.info("Forecasting complete!")
    LOGGER.info("=" * 80)
    
    return forecast_df


def predict_from_input(input_file='input/input_for_anomaly.py'):
    """
    Predict anomalies from user input file.
    
    Args:
        input_file: Path to input file with oxygen data
    """
    LOGGER.info("=" * 80)
    LOGGER.info("ANOMALY PREDICTION FROM INPUT FILE")
    LOGGER.info("=" * 80)
    
    # Load input data
    LOGGER.info("Loading input data", file=input_file)
    
    import sys
    import importlib.util
    
    spec = importlib.util.spec_from_file_location("input_module", input_file)
    input_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(input_module)
    
    # Check if CSV file is specified
    if hasattr(input_module, 'csv_file'):
        LOGGER.info("Loading data from CSV", path=input_module.csv_file)
        df = pd.read_csv(input_module.csv_file)
        df['time'] = pd.to_datetime(df['time'], format='mixed')
    elif hasattr(input_module, 'input_data'):
        LOGGER.info("Loading data from input_data variable")
        df = pd.DataFrame(input_module.input_data)
        df['time'] = pd.to_datetime(df['time'])
    else:
        LOGGER.error("No input data found. Please define 'input_data' or 'csv_file' in input file")
        return
    
    LOGGER.info("Data loaded", total_samples=len(df))
    
    # Load models
    LOGGER.info("Loading trained models")
    feature_path = Path("artifacts/anomaly_detector/feature_engineer.pkl")
    model_path = Path("artifacts/anomaly_detector/anomaly_detector.pkl")
    
    if not feature_path.exists() or not model_path.exists():
        LOGGER.error("Models not found. Please train models first: make train-detector")
        return
    
    feature_engineer = OxygenFeatureEngineer.load(str(feature_path))
    detector = UnifiedAnomalyDetector.load(str(model_path))
    LOGGER.info("Models loaded")
    
    # Make predictions
    LOGGER.info("Running anomaly detection")
    results = UnifiedAnomalyDetector.predict_single(
        oxygen_data=df,
        feature_engineer=feature_engineer,
        detector=detector,
        threshold=0.25
    )
    
    # Add results to dataframe
    df['anomaly_score'] = [r['anomaly_score'] for r in results]
    df['severity'] = [r['severity'] for r in results]
    df['severity_label'] = [r['label'] for r in results]
    df['is_anomaly'] = [r['is_anomaly'] for r in results]
    
    # Show statistics
    anomaly_count = sum(r['is_anomaly'] for r in results)
    anomaly_rate = anomaly_count / len(results) * 100
    
    LOGGER.info("=" * 80)
    LOGGER.info("RESULTS")
    LOGGER.info("=" * 80)
    LOGGER.info(
        "Summary",
        total_samples=len(df),
        anomalies_detected=anomaly_count,
        anomaly_rate=f"{anomaly_rate:.2f}%"
    )
    
    # Severity breakdown
    severity_counts = {
        'normal': sum(1 for r in results if r['severity'] == 0),
        'mild': sum(1 for r in results if r['severity'] == 1),
        'moderate': sum(1 for r in results if r['severity'] == 2),
        'severe': sum(1 for r in results if r['severity'] == 3)
    }
    LOGGER.info("Severity breakdown", distribution=severity_counts)
    
    # Save results
    output_path = Path("prediction_results.csv")
    df.to_csv(output_path, index=False)
    LOGGER.info("Results saved", path=str(output_path))
    
    # Show sample anomalies if any
    anomalies = df[df['is_anomaly']]
    if len(anomalies) > 0:
        LOGGER.info("Sample anomalies (showing first 10)")
        for i, (_, row) in enumerate(anomalies.head(10).iterrows()):
            LOGGER.info(
                f"Anomaly {i+1}",
                time=str(row['time']),
                oxygen=f"{row['Oxygen[%sat]']:.2f}%",
                score=f"{row['anomaly_score']:.3f}",
                severity=row['severity_label']
            )
    
    LOGGER.info("=" * 80)
    LOGGER.info("PREDICTION COMPLETE")
    LOGGER.info("=" * 80)
    
    return df


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description="OxyTrace - Oxygen Anomaly Detection & Forecasting")

    parser.add_argument("--demo", action="store_true", help="Run demo workflow")
    parser.add_argument("--predict", action="store_true", help="Predict anomalies from input file")
    parser.add_argument("--input-file", type=str, default="input/input_for_anomaly.py", 
                        help="Path to input file (default: input/input_for_anomaly.py)")
    parser.add_argument("--forecast", action="store_true", help="Generate forecasts (trains if needed)")
    parser.add_argument("--train", action="store_true", help="Force retrain forecaster before predicting")
    parser.add_argument(
        "--horizon", type=int, default=7, help="Forecast horizon in days (default: 7)"
    )
    parser.add_argument(
        "--data-percent", type=float, default=5.0, help="Percentage of data to use for demo (default: 5%%)"
    )

    args = parser.parse_args()

    if args.demo:
        demo_workflow(args.data_percent)
    elif args.predict:
        predict_from_input(args.input_file)
    elif args.forecast:
        train_and_forecast(args.horizon, train_first=args.train)
    else:
        # Default: show help
        LOGGER.info("OxyTrace - Oxygen Sensor Anomaly Detection & Forecasting")
        LOGGER.info("")
        LOGGER.info("Usage:")
        LOGGER.info("  python -m oxytrace.src.main --demo              # Run demo workflow")
        LOGGER.info("  python -m oxytrace.src.main --predict           # Predict from input/input_for_anomaly.py")
        LOGGER.info("  python -m oxytrace.src.main --predict --input-file path/to/file.py  # Custom input")
        LOGGER.info("  python -m oxytrace.src.main --forecast          # Generate forecast (uses existing model)")
        LOGGER.info("  python -m oxytrace.src.main --forecast --train  # Retrain & forecast")
        LOGGER.info("  python -m oxytrace.src.main --forecast --horizon 14  # 14-day forecast")
        LOGGER.info("  python oxytrace/src/train_detector.py           # Train anomaly detector")
        LOGGER.info("  python oxytrace/src/train_forecaster.py         # Train forecaster only")
        LOGGER.info("")
        LOGGER.info("Or use Makefile commands:")
        LOGGER.info("  make train      # Train models")
        LOGGER.info("  make predict    # Predict anomalies from input file")
        LOGGER.info("  make notebook   # Launch Jupyter")
        LOGGER.info("  make help       # Show all commands")


if __name__ == "__main__":
    main()
