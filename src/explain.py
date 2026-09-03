"""
Turn a risk score into something a plant manager can actually act on.

A floor manager does not trust, and should not have to trust, a bare number
like "0.87". They trust "this pump's tool wear and torque look like the
last 12 pumps that failed this way, inspect it this week." This module
uses SHAP to find WHY the model flagged a machine, then renders that as a
plain-English explanation grounded only in facts computed from the data
(no hallucinated reasoning) -- with an optional LLM pass to smooth the
prose, and a template fallback so the whole project runs with zero API keys.
"""
import os
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "failure_model.joblib"
DATA_PATH = ROOT / "data" / "ai4i2020.csv"

FEATURE_LABELS = {
    "Air temperature [K]": "air temperature",
    "Process temperature [K]": "process temperature",
    "Rotational speed [rpm]": "rotational speed",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "tool wear",
}


def _load():
    bundle = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    return bundle, df


def load_scored_fleet_ids(top_n: int = 5) -> list:
    """Convenience helper (used by tests/demos): the highest-risk Product IDs."""
    bundle, df = _load()
    pipe = bundle["pipeline"]
    num_feats, cat_feats = bundle["features_numeric"], bundle["features_categorical"]
    df = df.copy()
    df["p_fail"] = pipe.predict_proba(df[num_feats + cat_feats])[:, 1]
    return df.sort_values("p_fail", ascending=False)["Product ID"].head(top_n).tolist()


def _percentile_among_failures(df, feature, value):
    failed = df[df["Machine failure"] == 1][feature]
    if len(failed) == 0:
        return None
    return float((failed < value).mean() * 100)


def explain_machine(product_id: str, top_k: int = 2, use_llm: bool = False) -> dict:
    bundle, df = _load()
    pipe = bundle["pipeline"]
    num_feats, cat_feats = bundle["features_numeric"], bundle["features_categorical"]

    row = df[df["Product ID"] == product_id]
    if row.empty:
        raise ValueError(f"Unknown Product ID: {product_id}")
    row = row.iloc[[0]]
    X_row = row[num_feats + cat_feats]

    p_fail = float(pipe.predict_proba(X_row)[0, 1])

    # SHAP on the fitted XGBoost step, using the pipeline's own preprocessing
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    X_transformed = pre.transform(df[num_feats + cat_feats])
    X_row_transformed = pre.transform(X_row)
    feature_names = pre.get_feature_names_out()

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_row_transformed)
    shap_row = shap_values[0] if shap_values.ndim > 1 else shap_values

    order = np.argsort(-np.abs(shap_row))[:top_k]
    drivers = []
    for idx in order:
        raw_name = feature_names[idx]
        clean_name = raw_name.replace("num__", "").replace("cat__", "")
        contribution = float(shap_row[idx])
        detail = {"feature": clean_name, "shap_contribution": round(contribution, 4)}
        if clean_name in num_feats:
            value = float(row[clean_name].iloc[0])
            pct = _percentile_among_failures(df, clean_name, value)
            detail["value"] = value
            detail["percentile_vs_past_failures"] = round(pct, 1) if pct is not None else None
        drivers.append(detail)

    if p_fail >= 0.7:
        urgency = "Inspect within 24-48 hours"
    elif p_fail >= 0.3:
        urgency = "Schedule inspection this week"
    else:
        urgency = "Monitor; no immediate action required"

    plain_english = _render_template(product_id, p_fail, drivers, urgency)
    if use_llm and os.environ.get("ANTHROPIC_API_KEY"):
        plain_english = _polish_with_llm(product_id, p_fail, drivers, urgency, plain_english)

    return {
        "product_id": product_id,
        "estimated_failure_probability": round(p_fail, 4),
        "top_drivers": drivers,
        "recommended_action": urgency,
        "explanation": plain_english,
    }


def _render_template(product_id, p_fail, drivers, urgency) -> str:
    parts = []
    for d in drivers:
        label = FEATURE_LABELS.get(d["feature"], d["feature"])
        if "value" in d and d.get("percentile_vs_past_failures") is not None:
            parts.append(
                f"{label} is reading {d['value']:.1f}, higher than {d['percentile_vs_past_failures']:.0f}% "
                f"of machines that have actually failed before"
            )
        else:
            parts.append(f"{label} is contributing to the risk score")
    driver_text = " and ".join(parts) if parts else "an unusual combination of sensor readings"
    return (
        f"Machine {product_id} is flagged at {p_fail:.0%} estimated failure risk. "
        f"Main reason: {driver_text}, a pattern consistent with prior failures in this fleet. "
        f"{urgency}."
    )


def _polish_with_llm(product_id, p_fail, drivers, urgency, fallback_text) -> str:
    """Optional: rewrite the template explanation in more natural prose using an
    LLM, strictly grounded in the facts already computed above (no new claims).
    Falls back silently to the template text if no API key or the call fails,
    so the project always runs without requiring a paid key."""
    try:
        import anthropic

        client = anthropic.Anthropic()
        facts = json.dumps(
            {"product_id": product_id, "p_fail": p_fail, "drivers": drivers, "urgency": urgency}
        )
        msg = client.messages.create(
            model="claude-3-5-haiku-latest",
            max_tokens=150,
            messages=[{
                "role": "user",
                "content": (
                    "Rewrite this maintenance alert for a plant floor manager with no data "
                    "science background, in 2 short sentences. Use ONLY the facts given, "
                    "do not invent numbers or causes.\n\nFacts: " + facts
                ),
            }],
        )
        return msg.content[0].text.strip()
    except Exception:
        return fallback_text


if __name__ == "__main__":
    bundle, df = _load()
    # Demo on a handful of the highest-risk machines
    pipe = bundle["pipeline"]
    num_feats, cat_feats = bundle["features_numeric"], bundle["features_categorical"]
    df["p_fail"] = pipe.predict_proba(df[num_feats + cat_feats])[:, 1]
    sample_ids = df.sort_values("p_fail", ascending=False)["Product ID"].head(3).tolist()
    for pid in sample_ids:
        result = explain_machine(pid)
        print(json.dumps(result, indent=2))
        print("-" * 60)
