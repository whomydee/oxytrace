"""
Training script for oxygen forecasting model
"""

import sys
from pathlib import Path

import pandas as pd

from oxytrace.core.models.forecaster import Forecaster

sys.path.insert(0, str(Path(__file__).parent))


def load_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    df["time"] = pd.to_datetime(df["time"], format="mixed")
    df = df[df["Oxygen[%sat]"].notna()].copy()
    df = df.sort_values("time").reset_index(drop=True)
    print(f"Loaded {len(df):,} oxygen readings")
    return df


def train_forecaster(df, output_dir="artifacts/forecaster"):
    print("\n" + "=" * 60)
    print("TRAINING OXYGEN FORECASTER")
    print("=" * 60)

    print("\nStep 1: Training forecaster...")
    forecaster = Forecaster(short_window=1440, long_window=10080, alpha=0.3)

    forecaster.fit(df)

    print("Stats:")
    print(f"  Mean: {forecaster.global_mean_:.2f}%")
    print(f"  Std: {forecaster.global_std_:.2f}%")
    print(f"  Trend: {forecaster.trend_:.6f}%/min ({forecaster.trend_ * 1440:.3f}%/day)")
    print(f"  Seasonality: {forecaster.has_seasonality_}")
    if forecaster.has_seasonality_:
        print(f"  Daily range: {forecaster.daily_pattern_.min():.2f}% to {forecaster.daily_pattern_.max():.2f}%")
        if forecaster.weekly_pattern_ is not None:
            print(f"  Weekly range: {forecaster.weekly_pattern_.min():.2f}% to {forecaster.weekly_pattern_.max():.2f}%")
    print(f"  Residual std: {forecaster.residual_std_:.2f}%")

    print("\nStep 2: Saving model...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    forecaster.save(output_path / "oxygen_forecaster.pkl")
    print(f"Saved to {output_path}")

    return forecaster


def main():
    df = load_data("oxytrace/data/dataset.csv")
    train_forecaster(df)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print("artifacts/forecaster/oxygen_forecaster.pkl")


if __name__ == "__main__":
    main()
