import pandas as pd


def evaluate_at_threshold(df, threshold):
    """Evaluate model performance at a specific threshold."""
    true_anomalies = df["anomaly_type"] != "normal"
    predicted_anomalies = df["anomaly_score"] >= threshold

    tp = (true_anomalies & predicted_anomalies).sum()
    fp = (~true_anomalies & predicted_anomalies).sum()
    # tn = (~true_anomalies & ~predicted_anomalies).sum()
    fn = (true_anomalies & ~predicted_anomalies).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {"threshold": threshold, "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def main():
    df = pd.read_csv("analysis_data/synthetic_test_results.csv")

    print("=" * 80)
    print("THRESHOLD ANALYSIS")
    print("=" * 80)

    thresholds = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]
    results = []

    for threshold in thresholds:
        result = evaluate_at_threshold(df, threshold)
        results.append(result)
        print(f"\nThreshold {threshold:.2f}:")
        print(f"  Precision: {result['precision']:.3f}")
        print(f"  Recall:    {result['recall']:.3f}")
        print(f"  F1-Score:  {result['f1']:.3f}")
        print(f"  TP: {result['tp']:4d}  FP: {result['fp']:4d}  FN: {result['fn']:4d}")

    # Find best F1
    best = max(results, key=lambda x: x["f1"])
    print("\n" + "=" * 80)
    print(f"BEST F1-SCORE: {best['f1']:.3f} at threshold {best['threshold']:.2f}")
    print(f"  Precision: {best['precision']:.3f}")
    print(f"  Recall:    {best['recall']:.3f}")
    print("=" * 80)

    # Per-anomaly-type breakdown at best threshold
    print(f"\nDETECTION BY TYPE (threshold={best['threshold']:.2f}):")
    print("-" * 80)
    for anom_type in ["point_anomaly", "collective_anomaly", "stuck_sensor", "noisy_sensor"]:
        subset = df[df["anomaly_type"] == anom_type]
        detected = (subset["anomaly_score"] >= best["threshold"]).sum()
        recall = detected / len(subset)
        mean_score = subset["anomaly_score"].mean()
        print(f"{anom_type:20s}: {detected:3d}/{len(subset):3d} ({recall:5.1%})  mean_score={mean_score:.3f}")


if __name__ == "__main__":
    main()
