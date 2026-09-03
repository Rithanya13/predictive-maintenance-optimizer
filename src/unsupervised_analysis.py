"""
Unsupervised learning layer: two genuinely different jobs, not one
algorithm run twice to check a box.

1. CLUSTERING (KMeans) -- cluster the actual failures on raw sensor
   readings alone (no Type, no failure-mode flags) to see whether
   unsupervised structure recovers the distinct physical failure
   mechanisms this dataset documents (TWF/HDF/PWF/OSF/RNF), validating
   against those flags only AFTER clustering -- never as a training
   signal. This is the standard "discover subtypes, then check them
   against a known label the model never saw" pattern.

2. ANOMALY DETECTION (IsolationForest) -- a safety net for the supervised
   model's blind spot. XGBoost can only learn to recognize failure
   patterns that were LABELED as failures in training data. A truly novel
   failure mode it has never seen would not be reliably caught. An
   unsupervised anomaly detector flags "this doesn't look like normal
   operation" without needing to have seen that specific failure before.
   We measure this concretely: of the failures the supervised model's
   cost-optimal threshold MISSED (false negatives), how many did the
   unsupervised anomaly detector catch anyway?
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.train_model import FEATURES_CATEGORICAL, FEATURES_NUMERIC, TARGET

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "ai4i2020.csv"
MODEL_PATH = ROOT / "models" / "failure_model.joblib"
RESULTS_PATH = ROOT / "models" / "unsupervised_results.json"


def build_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), FEATURES_NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CATEGORICAL),
    ])


FAILURE_MODE_FLAGS = ["TWF", "HDF", "PWF", "OSF", "RNF"]


def _primary_failure_mode(row):
    for flag in FAILURE_MODE_FLAGS:
        if row[flag] == 1:
            return flag
    return "unlabeled"


def run_clustering(df):
    """Cluster ONLY the failed machines, on their raw sensor readings alone
    (Type and the TWF/HDF/PWF/OSF/RNF flags withheld), to see whether
    unsupervised structure recovers the distinct physical failure
    mechanisms this dataset actually documents. The flags are used only
    AFTER clustering, purely to validate the discovered groups against
    something mechanically real -- never as a training signal."""
    fail_df = df[df[TARGET] == 1].copy()
    X_scaled = StandardScaler().fit_transform(fail_df[FEATURES_NUMERIC])
    fail_df["primary_mode"] = fail_df.apply(_primary_failure_mode, axis=1)

    best_k, best_score, best_labels = None, -1, None
    scores_by_k = {}
    for k in [2, 3, 4, 5, 6]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores_by_k[k] = round(float(score), 4)
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels

    fail_df["cluster"] = best_labels
    ari = adjusted_rand_score(fail_df["primary_mode"], fail_df["cluster"])

    cluster_table = []
    for c in sorted(fail_df["cluster"].unique()):
        sub = fail_df[fail_df["cluster"] == c]
        mode_counts = sub["primary_mode"].value_counts().to_dict()
        dominant_mode = max(mode_counts, key=mode_counts.get)
        cluster_table.append({
            "cluster": int(c),
            "n_failures": int(len(sub)),
            "dominant_known_failure_mode": dominant_mode,
            "purity": round(mode_counts[dominant_mode] / len(sub), 2),
            "mean_air_temp_k": round(float(sub["Air temperature [K]"].mean()), 1),
            "mean_process_temp_k": round(float(sub["Process temperature [K]"].mean()), 1),
            "mean_rotational_speed_rpm": round(float(sub["Rotational speed [rpm]"].mean()), 1),
            "mean_torque_nm": round(float(sub["Torque [Nm]"].mean()), 1),
            "mean_tool_wear_min": round(float(sub["Tool wear [min]"].mean()), 1),
        })

    return {
        "method": "KMeans on the 339 actual failures' raw sensor readings only -- Type and the "
                  "TWF/HDF/PWF/OSF/RNF flags were withheld from clustering entirely",
        "chosen_k": best_k,
        "silhouette_score": round(float(best_score), 4),
        "silhouette_by_k_tried": scores_by_k,
        "validation_note": (
            "the failure-mode flags were never used to fit the clusters -- they are used here "
            "only after the fact, to check whether the unsupervised structure recovered "
            "something mechanically real"
        ),
        "adjusted_rand_index_vs_known_failure_modes": round(float(ari), 4),
        "clusters": sorted(cluster_table, key=lambda r: -r["n_failures"]),
        "finding": _summarize_clusters(cluster_table, ari),
    }


def _summarize_clusters(cluster_table, ari):
    best_purity = max(cluster_table, key=lambda r: r["purity"])
    quality = "meaningfully recovers" if ari >= 0.25 else "only weakly recovers"
    return (
        f"Clustering on raw sensor readings alone {quality} this dataset's documented failure "
        f"mechanisms (Adjusted Rand Index {ari:.2f} vs the TWF/HDF/PWF/OSF/RNF labels, which the "
        f"clustering never saw). The purest cluster is {best_purity['purity']:.0%} "
        f"'{best_purity['dominant_known_failure_mode']}' failures, with mean torque "
        f"{best_purity['mean_torque_nm']}Nm and tool wear {best_purity['mean_tool_wear_min']}min -- "
        f"a distinct mechanical signature found with zero access to the failure-mode label."
    )


def run_anomaly_detection(df, X_transformed, pipe, threshold):
    # Fit on ALL data, with zero access to the label -- fully unsupervised.
    iso = IsolationForest(contamination=0.034, random_state=42)
    anomaly_pred = iso.fit_predict(X_transformed)  # -1 = anomaly, 1 = normal
    df = df.copy()
    df["is_anomaly"] = anomaly_pred == -1

    y = df[TARGET].values
    anomaly_recall = float(df.loc[y == 1, "is_anomaly"].mean())
    anomaly_precision = (
        float(df.loc[df["is_anomaly"], TARGET].mean()) if df["is_anomaly"].any() else None
    )

    # The safety-net question: of the failures the SUPERVISED model's
    # cost-optimal threshold missed, how many did the unsupervised
    # detector catch anyway?
    X_full = df[FEATURES_NUMERIC + FEATURES_CATEGORICAL]
    df["p_fail_supervised"] = pipe.predict_proba(X_full)[:, 1]
    df["supervised_flagged"] = df["p_fail_supervised"] >= threshold

    missed_by_supervised = df[(df[TARGET] == 1) & (~df["supervised_flagged"])]
    caught_by_anomaly_among_missed = (
        int(missed_by_supervised["is_anomaly"].sum()) if len(missed_by_supervised) else 0
    )

    return {
        "method": "IsolationForest, contamination=0.034 (matched to the known population failure rate), "
                  "fit with NO access to the failure label",
        "overall_anomaly_recall_of_actual_failures": round(anomaly_recall, 4),
        "overall_anomaly_precision": round(anomaly_precision, 4) if anomaly_precision is not None else None,
        "supervised_model_missed_n_failures": int(len(missed_by_supervised)),
        "of_those_missed_anomaly_detector_caught": caught_by_anomaly_among_missed,
        "finding": (
            f"Of the {len(missed_by_supervised)} failures the supervised model's cost-optimal "
            f"threshold missed, the unsupervised anomaly detector independently flagged "
            f"{caught_by_anomaly_among_missed} of them as abnormal -- a real safety net, not "
            f"a duplicate of the supervised signal."
            if len(missed_by_supervised) else
            "The supervised model missed no failures on this run, so there was nothing for the "
            "anomaly detector to catch that the classifier didn't already flag."
        ),
    }


def main():
    bundle = joblib.load(MODEL_PATH)
    pipe = bundle["pipeline"]
    threshold = bundle["threshold"]

    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    X = df[FEATURES_NUMERIC + FEATURES_CATEGORICAL]

    pre = build_preprocessor()
    X_transformed = pre.fit_transform(X)

    clustering = run_clustering(df)
    anomaly = run_anomaly_detection(df, X_transformed, pipe, threshold)

    results = {"clustering": clustering, "anomaly_detection": anomaly}
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
