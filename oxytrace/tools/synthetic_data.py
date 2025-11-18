"""
Generate synthetic anomaly data for testing

Creates datasets with different types of anomalies to verify the model works.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def create_normal_data(n_samples=10000, mean=87, std=2):
    """Create normal oxygen readings with some natural variation."""
    np.random.seed(42)

    time = pd.date_range("2024-01-01", periods=n_samples, freq="1min")
    oxygen = np.random.normal(mean, std, n_samples)

    # Add some diurnal pattern (oxygen varies slightly through the day)
    hours = time.hour
    diurnal = 2 * np.sin(2 * np.pi * hours / 24)
    oxygen += diurnal

    df = pd.DataFrame({"time": time, "Oxygen[%sat]": oxygen, "anomaly_type": "normal"})

    return df


def inject_point_anomalies(df, n_anomalies=50):
    """Inject point anomalies (sudden spikes/drops)."""
    indices = np.random.choice(len(df), n_anomalies, replace=False)

    for idx in indices:
        # Randomly spike or drop
        if np.random.rand() > 0.5:
            df.loc[idx, "Oxygen[%sat]"] += np.random.uniform(15, 30)  # Spike
        else:
            df.loc[idx, "Oxygen[%sat]"] -= np.random.uniform(15, 30)  # Drop

        df.loc[idx, "anomaly_type"] = "point_anomaly"

    return df


def inject_collective_anomalies(df, n_events=3, duration=100):
    """Inject collective anomalies (gradual drift)."""
    for _ in range(n_events):
        start_idx = np.random.randint(0, len(df) - duration)
        end_idx = start_idx + duration

        # Gradual increase or decrease
        drift = np.linspace(0, np.random.uniform(5, 10), end_idx - start_idx)
        if np.random.rand() > 0.5:
            drift = -drift

        df.loc[start_idx : end_idx - 1, "Oxygen[%sat]"] += drift
        df.loc[start_idx : end_idx - 1, "anomaly_type"] = "collective_anomaly"

    return df


def inject_stuck_sensor(df, n_events=2, duration=50):
    """Inject stuck sensor anomalies (same value repeated)."""
    for _ in range(n_events):
        start_idx = np.random.randint(0, len(df) - duration)
        end_idx = start_idx + duration

        stuck_value = df.loc[start_idx, "Oxygen[%sat]"]
        df.loc[start_idx : end_idx - 1, "Oxygen[%sat]"] = stuck_value
        df.loc[start_idx : end_idx - 1, "anomaly_type"] = "stuck_sensor"

    return df


def inject_noisy_sensor(df, n_events=2, duration=100):
    """Inject noisy sensor periods (high variance)."""
    for _ in range(n_events):
        start_idx = np.random.randint(0, len(df) - duration)
        end_idx = start_idx + duration

        noise = np.random.normal(0, 5, end_idx - start_idx)
        df.loc[start_idx : end_idx - 1, "Oxygen[%sat]"] += noise
        df.loc[start_idx : end_idx - 1, "anomaly_type"] = "noisy_sensor"

    return df


def main():
    print("=" * 60)
    print("GENERATING SYNTHETIC ANOMALY DATA")
    print("=" * 60)

    # Create base normal data
    print("\n Creating normal baseline data...")
    df = create_normal_data(n_samples=10000)

    # Inject different anomaly types
    print("Injecting point anomalies...")
    df = inject_point_anomalies(df, n_anomalies=50)

    print("Injecting collective anomalies...")
    df = inject_collective_anomalies(df, n_events=3, duration=100)

    print("Injecting stuck sensor anomalies...")
    df = inject_stuck_sensor(df, n_events=2, duration=50)

    print("Injecting noisy sensor anomalies...")
    df = inject_noisy_sensor(df, n_events=2, duration=100)

    # Save
    output_path = Path("analysis_data/synthetic_data")
    output_path.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path / "synthetic_anomalies.csv", index=False)

    # Summary
    print("\n" + "=" * 60)
    print("SYNTHETIC DATA SUMMARY")
    print("=" * 60)
    print(f"\nTotal samples: {len(df):,}")
    print("Anomaly type distribution:")
    print(df["anomaly_type"].value_counts())

    print(f"\nData saved to: {output_path / 'synthetic_anomalies.csv'}")


if __name__ == "__main__":
    main()
