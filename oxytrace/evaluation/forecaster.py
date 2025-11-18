import sys
from pathlib import Path

import numpy as np
import pandas as pd

from oxytrace.core.models.forecaster import Forecaster

sys.path.insert(0, str(Path(__file__).parent))


def load_data(filepath):
    df = pd.read_csv(filepath)
    df["time"] = pd.to_datetime(df["time"], format="mixed")
    df = df[df["Oxygen[%sat]"].notna()].copy()
    df = df.sort_values("time").reset_index(drop=True)
    return df


def evaluate_metrics(actual, predicted):
    errors = actual - predicted
    return {
        "MAE": np.abs(errors).mean(),
        "RMSE": np.sqrt((errors**2).mean()),
        "MAPE": (np.abs(errors / actual) * 100).mean(),
    }


def walk_forward_validation(df, n_splits=5, forecast_horizon=10080):
    print("=" * 60)
    print("WALK-FORWARD VALIDATION")
    print("=" * 60)

    total_samples = len(df)
    split_size = total_samples // (n_splits + 1)
    all_results = []

    for i in range(n_splits):
        print(f"\n--- Split {i+1}/{n_splits} ---")

        train_end = split_size * (i + 1)
        test_end = min(train_end + forecast_horizon, total_samples)
        actual_horizon = test_end - train_end

        train_df = df.iloc[:train_end]
        test_df = df.iloc[train_end:test_end]

        if len(test_df) < 100:
            print(f"Skipping - insufficient test data ({len(test_df)} samples)")
            continue

        print(f"Training on {len(train_df):,} samples")
        print(f"Testing on {len(test_df):,} samples ({actual_horizon} minutes)")

        # Train forecaster
        forecaster = Forecaster()
        forecaster.fit(train_df)

        # Make predictions
        forecast_df = forecaster.predict(horizon=actual_horizon, return_uncertainty=True)

        # Compare to actual
        actual_values = test_df["Oxygen[%sat]"].values
        predicted_values = forecast_df["predicted_oxygen"].values[: len(actual_values)]

        metrics = evaluate_metrics(actual_values, predicted_values)

        print(f"MAE:  {metrics['MAE']:.3f}%")
        print(f"RMSE: {metrics['RMSE']:.3f}%")
        print(f"MAPE: {metrics['MAPE']:.2f}%")

        all_results.append(metrics)

    # Average across splits
    print("\n" + "=" * 60)
    print("AVERAGE PERFORMANCE")
    print("=" * 60)

    avg_mae = np.mean([r["MAE"] for r in all_results])
    avg_rmse = np.mean([r["RMSE"] for r in all_results])
    avg_mape = np.mean([r["MAPE"] for r in all_results])

    print(f"Average MAE:  {avg_mae:.3f}%")
    print(f"Average RMSE: {avg_rmse:.3f}%")
    print(f"Average MAPE: {avg_mape:.2f}%")

    return all_results


def test_forecast_horizons(df):
    print("\n" + "=" * 60)
    print("FORECAST HORIZON ANALYSIS")
    print("=" * 60)

    split_idx = int(len(df) * 0.8)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    # Train forecaster
    forecaster = Forecaster()
    forecaster.fit(train_df)

    # Test different horizons
    horizons = [1440, 2880, 5040, 7200, 10080]  # 1, 2, 3.5, 5, 7 days
    horizon_names = ["1 day", "2 days", "3.5 days", "5 days", "7 days"]

    print(f"\nTrained on {len(train_df):,} samples")
    print(f"Testing on {len(test_df):,} samples\n")

    for horizon, name in zip(horizons, horizon_names):
        if horizon > len(test_df):
            continue

        forecast_df = forecaster.predict(horizon=horizon, return_uncertainty=True)
        actual_values = test_df["Oxygen[%sat]"].values[:horizon]
        predicted_values = forecast_df["predicted_oxygen"].values

        metrics = evaluate_metrics(actual_values, predicted_values)

        print(f"{name:12s}: MAE={metrics['MAE']:.3f}%  RMSE={metrics['RMSE']:.3f}%  MAPE={metrics['MAPE']:.2f}%")


def analyze_uncertainty(df):
    print("\n" + "=" * 60)
    print("UNCERTAINTY CALIBRATION")
    print("=" * 60)

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    forecaster = Forecaster()
    forecaster.fit(train_df)

    horizon = min(10080, len(test_df))
    forecast_df = forecaster.predict(horizon=horizon, return_uncertainty=True)

    actual_values = test_df["Oxygen[%sat]"].values[:horizon]
    lower_bounds = forecast_df["lower_bound_95"].values
    upper_bounds = forecast_df["upper_bound_95"].values

    within_bounds = (actual_values >= lower_bounds) & (actual_values <= upper_bounds)
    coverage = within_bounds.mean() * 100

    print(f"\n95% Confidence Interval Coverage: {coverage:.1f}%")
    print("(Expected: 95%, Good if: 90-98%)")

    if coverage < 90:
        print("⚠️  Uncertainty underestimated - intervals too narrow")
    elif coverage > 98:
        print("⚠️  Uncertainty overestimated - intervals too wide")
    else:
        print("✓ Uncertainty well-calibrated")


def main():
    df = load_data("oxytrace/dataset/dataset.csv")

    walk_forward_validation(df, n_splits=5, forecast_horizon=10080)
    test_forecast_horizons(df)
    analyze_uncertainty(df)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
