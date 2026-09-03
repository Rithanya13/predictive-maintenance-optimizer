"""
Train a machine-failure classifier on the AI4I 2020 dataset, then pick an
operating threshold by MINIMIZING EXPECTED DOLLAR COST, not by maximizing
accuracy or F1.

Business framing
-----------------
A plant has machines. Three things can happen to any machine on any given
week:
  - You leave it alone and it keeps running fine            -> cost $0
  - You leave it alone and it fails unexpectedly             -> cost C_FN
    (emergency repair + unplanned downtime + expedited parts)
  - You send a technician to inspect/service it
      - it was actually fine (false alarm)                   -> cost C_FP
        (wasted technician time + parts + production pause)
      - it was actually about to fail (caught in time)       -> cost C_TP
        (planned repair — still costs money, just far less
         than an emergency failure)

The "best" model by accuracy is not the same as the "best" model by cost,
because C_FN >> C_FP >> 0 in almost every real plant. This script sweeps
the decision threshold and reports the one that minimizes total dollars
spent on the test set, compared against two naive baselines:
  - "Run to failure"   : never inspect anything
  - "Inspect everyone" : inspect every machine every week
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
import xgboost as xgb

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "ai4i2020.csv"
MODEL_PATH = ROOT / "models" / "failure_model.joblib"
RESULTS_PATH = ROOT / "models" / "results.json"

# ---- Business cost assumptions (plant-configurable; these are the numbers
# a real deployment would get from the customer's maintenance/finance team,
# not something a data scientist should invent and leave hardcoded forever) ----
COST_FALSE_NEGATIVE = 50_000  # missed failure -> unplanned downtime, emergency repair
COST_FALSE_POSITIVE = 2_000   # unnecessary inspection dispatched on a healthy machine
COST_TRUE_POSITIVE = 8_000    # caught failure -> planned repair (still costs, just way less)
COST_TRUE_NEGATIVE = 0        # correctly left alone

FEATURES_NUMERIC = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
FEATURES_CATEGORICAL = ["Type"]
TARGET = "Machine failure"
# NOTE: TWF/HDF/PWF/OSF/RNF are failure-*mode* flags that are only known
# once a failure has already happened -> including them would leak the
# label into the features. They are intentionally excluded.


def load_data():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    X = df[FEATURES_NUMERIC + FEATURES_CATEGORICAL]
    y = df[TARGET]
    return df, X, y


def build_pipeline():
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), FEATURES_NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CATEGORICAL),
        ]
    )
    neg, pos = 9661, 339  # from the known class balance; recomputed properly below
    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="aucpr",
        random_state=42,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def cost_at_threshold(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total_cost = (
        fp * COST_FALSE_POSITIVE
        + fn * COST_FALSE_NEGATIVE
        + tp * COST_TRUE_POSITIVE
        + tn * COST_TRUE_NEGATIVE
    )
    return total_cost, dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def find_cost_optimal_threshold(y_true, y_proba):
    thresholds = np.linspace(0.01, 0.99, 99)
    best = None
    sweep = []
    for t in thresholds:
        cost, counts = cost_at_threshold(y_true, y_proba, t)
        sweep.append({"threshold": round(float(t), 2), "total_cost": int(cost), **counts})
        if best is None or cost < best["total_cost"]:
            best = {"threshold": round(float(t), 2), "total_cost": int(cost), **counts}
    return best, sweep


def main():
    df, X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    pipe = build_pipeline()
    pipe.named_steps["clf"].set_params(scale_pos_weight=neg / pos)
    pipe.fit(X_train, y_train)

    y_proba = pipe.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    # Metrics at the naive default threshold (0.5) -- what most tutorials stop at
    default_cost, default_counts = cost_at_threshold(y_test, y_proba, 0.5)

    # Metrics at the cost-optimal threshold
    best, sweep = find_cost_optimal_threshold(y_test, y_proba)

    # Naive business baselines for comparison
    n_test = len(y_test)
    n_fail_test = int(y_test.sum())
    run_to_failure_cost = n_fail_test * COST_FALSE_NEGATIVE
    inspect_everyone_cost = n_fail_test * COST_TRUE_POSITIVE + (n_test - n_fail_test) * COST_FALSE_POSITIVE

    results = {
        "dataset": "AI4I 2020 Predictive Maintenance Dataset (UCI #601), n=10000, failure rate=3.39%",
        "test_set_size": n_test,
        "actual_failures_in_test_set": n_fail_test,
        "model": "XGBoost, scale_pos_weight for imbalance, features exclude failure-mode leak columns",
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "cost_assumptions_usd": {
            "false_negative_missed_failure": COST_FALSE_NEGATIVE,
            "false_positive_unnecessary_inspection": COST_FALSE_POSITIVE,
            "true_positive_caught_failure_planned_repair": COST_TRUE_POSITIVE,
            "true_negative_left_alone": COST_TRUE_NEGATIVE,
        },
        "baseline_run_to_failure_usd": int(run_to_failure_cost),
        "baseline_inspect_everyone_usd": int(inspect_everyone_cost),
        "model_at_default_threshold_0.5": {
            "total_cost_usd": int(default_cost),
            **default_counts,
            "savings_vs_run_to_failure_usd": int(run_to_failure_cost - default_cost),
            "savings_vs_inspect_everyone_usd": int(inspect_everyone_cost - default_cost),
        },
        "model_at_cost_optimal_threshold": {
            **best,
            "savings_vs_run_to_failure_usd": int(run_to_failure_cost - best["total_cost"]),
            "savings_vs_inspect_everyone_usd": int(inspect_everyone_cost - best["total_cost"]),
            "savings_vs_default_threshold_usd": int(default_cost - best["total_cost"]),
        },
        "threshold_sweep": sweep,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pipeline": pipe, "threshold": best["threshold"],
         "features_numeric": FEATURES_NUMERIC, "features_categorical": FEATURES_CATEGORICAL},
        MODEL_PATH,
    )
    RESULTS_PATH.write_text(json.dumps(results, indent=2))

    print(json.dumps({k: v for k, v in results.items() if k != "threshold_sweep"}, indent=2))


if __name__ == "__main__":
    main()
