"""
Segment-level generalization analysis.

IMPORTANT FRAMING: AI4I 2020's "Type" field (L/M/H) is a PRODUCT-QUALITY
VARIANT, not a real customer site. There is no real multi-customer
deployment data available in this public dataset. We use L/M/H here as a
stand-in for "different deployment segments" to demonstrate a real and
common enterprise-ML problem: a single global model, evaluated with a
single global threshold, does not perform identically across every
subgroup it's deployed against. Do not present this as "tested across
real customer sites" -- it is a segment-level generalization analysis
using the only sub-population structure this public dataset actually has.

Why this matters in practice: a platform that deploys the same trained
model across many customers (or, within one customer, across many product
lines / plants / equipment classes) will see the failure base rate and the
feature distributions shift by segment. A threshold picked to minimize
cost on the whole population is not guaranteed to be cost-optimal for
each individual segment -- this script measures exactly how much is left
on the table by NOT recalibrating per segment, and flags where a segment
has too few observed failures for recalibration to be statistically
trustworthy (rather than presenting a noisy result as if it were solid).
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.train_model import (
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
    COST_TRUE_POSITIVE,
    FEATURES_CATEGORICAL,
    FEATURES_NUMERIC,
    TARGET,
    cost_at_threshold,
    find_cost_optimal_threshold,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "ai4i2020.csv"
MODEL_PATH = ROOT / "models" / "failure_model.joblib"
RESULTS_PATH = ROOT / "models" / "segment_results.json"

# Below this many actual failures in a segment's test slice, a per-segment
# threshold sweep is too noisy to trust -- flag it rather than report it
# as if it were reliable.
MIN_FAILURES_FOR_RELIABLE_RECALIBRATION = 20


def main():
    bundle = joblib.load(MODEL_PATH)
    pipe = bundle["pipeline"]
    global_threshold = bundle["threshold"]

    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    X = df[FEATURES_NUMERIC + FEATURES_CATEGORICAL]
    y = df[TARGET]

    # Same split as train_model.py so this is the identical held-out test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    test_df = X_test.copy()
    test_df[TARGET] = y_test
    test_df["p_fail"] = pipe.predict_proba(X_test)[:, 1]

    segment_results = {}
    total_cost_at_global_threshold = 0
    total_cost_at_recalibrated_thresholds = 0

    for segment in sorted(test_df["Type"].unique()):
        seg = test_df[test_df["Type"] == segment]
        y_seg = seg[TARGET].values
        p_seg = seg["p_fail"].values
        n_failures = int(y_seg.sum())

        global_cost, global_counts = cost_at_threshold(y_seg, p_seg, global_threshold)
        recall_at_global = global_counts["tp"] / n_failures if n_failures else None
        fpr_at_global = (
            global_counts["fp"] / (global_counts["fp"] + global_counts["tn"])
            if (global_counts["fp"] + global_counts["tn"]) else None
        )

        reliable = n_failures >= MIN_FAILURES_FOR_RELIABLE_RECALIBRATION
        if reliable:
            best, _ = find_cost_optimal_threshold(y_seg, p_seg)
            recalibrated_threshold = best["threshold"]
            recalibrated_cost = best["total_cost"]
            recall_at_recalibrated = best["tp"] / n_failures if n_failures else None
            fpr_at_recalibrated = (
                best["fp"] / (best["fp"] + best["tn"]) if (best["fp"] + best["tn"]) else None
            )
        else:
            recalibrated_threshold = global_threshold
            recalibrated_cost = global_cost
            recall_at_recalibrated = recall_at_global
            fpr_at_recalibrated = fpr_at_global

        total_cost_at_global_threshold += global_cost
        total_cost_at_recalibrated_thresholds += recalibrated_cost

        segment_results[segment] = {
            "n_machines_in_test_set": int(len(seg)),
            "actual_failure_rate": round(float(y_seg.mean()), 4),
            "n_actual_failures": n_failures,
            "reliable_for_recalibration": reliable,
            "reliability_note": (
                f"only {n_failures} actual failures in this segment's test slice; "
                f"per-segment threshold not recalibrated (kept at the global threshold) "
                f"to avoid overfitting to noise"
                if not reliable else
                f"{n_failures} actual failures -- enough to recalibrate with reasonable confidence"
            ),
            "at_global_threshold": {
                "threshold": global_threshold,
                "total_cost_usd": int(global_cost),
                "recall": round(recall_at_global, 4) if recall_at_global is not None else None,
                "false_positive_rate": round(fpr_at_global, 4) if fpr_at_global is not None else None,
                **global_counts,
            },
            "at_segment_recalibrated_threshold": {
                "threshold": recalibrated_threshold,
                "total_cost_usd": int(recalibrated_cost),
                "recall": round(recall_at_recalibrated, 4) if recall_at_recalibrated is not None else None,
                "false_positive_rate": round(fpr_at_recalibrated, 4) if fpr_at_recalibrated is not None else None,
            },
            "savings_from_recalibration_usd": int(global_cost - recalibrated_cost),
        }

    summary = {
        "framing": (
            "L/M/H are AI4I 2020 product-quality variants, used here as a stand-in for "
            "distinct deployment segments -- NOT real customer sites. This is a "
            "segment-level generalization analysis, not a multi-customer field study."
        ),
        "global_threshold_used_as_baseline": global_threshold,
        "total_cost_if_one_global_threshold_everywhere_usd": int(total_cost_at_global_threshold),
        "total_cost_if_recalibrated_per_reliable_segment_usd": int(total_cost_at_recalibrated_thresholds),
        "total_savings_from_segment_recalibration_usd": int(
            total_cost_at_global_threshold - total_cost_at_recalibrated_thresholds
        ),
        "segments": segment_results,
    }

    RESULTS_PATH.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
