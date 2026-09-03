"""
FastAPI service exposing the pieces as one product:
  POST /predict     -> failure risk for a machine's current sensor readings
  GET  /schedule    -> this week's optimized maintenance schedule
  GET  /explain/{product_id} -> plain-English reason a machine was flagged
  GET  /segments    -> per-segment (product-type) generalization analysis
  GET  /unsupervised -> clustering + anomaly-detection findings
  GET  /health
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.explain import explain_machine
from src.schedule_optimizer import load_scored_fleet, solve_schedule, WEEKLY_CAPACITY_HOURS
import json
import joblib
from pathlib import Path as _Path

RESULTS_DIR = _Path(__file__).resolve().parent.parent / "models"

MODEL_PATH = _Path(__file__).resolve().parent.parent / "models" / "failure_model.joblib"

app = FastAPI(
    title="Predictive Maintenance Decision Service",
    description=(
        "Not just a risk score: predicts machine failure risk, converts it into "
        "a dollar-cost-optimal decision threshold, turns flagged machines into an "
        "actual technician schedule under an hours budget, and explains each flag "
        "in plain English for a non-technical plant manager."
    ),
    version="1.0.0",
)

_bundle = None


def get_bundle():
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


class MachineReading(BaseModel):
    air_temperature_k: float = Field(..., json_schema_extra={"example": 298.5})
    process_temperature_k: float = Field(..., json_schema_extra={"example": 309.0})
    rotational_speed_rpm: float = Field(..., json_schema_extra={"example": 1500})
    torque_nm: float = Field(..., json_schema_extra={"example": 45.0})
    tool_wear_min: float = Field(..., json_schema_extra={"example": 180})
    type: str = Field(..., json_schema_extra={"example": "M"}, description="Product quality variant: L, M, or H")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(reading: MachineReading):
    import pandas as pd

    bundle = get_bundle()
    pipe = bundle["pipeline"]
    threshold = bundle["threshold"]

    row = pd.DataFrame([{
        "Air temperature [K]": reading.air_temperature_k,
        "Process temperature [K]": reading.process_temperature_k,
        "Rotational speed [rpm]": reading.rotational_speed_rpm,
        "Torque [Nm]": reading.torque_nm,
        "Tool wear [min]": reading.tool_wear_min,
        "Type": reading.type,
    }])
    p_fail = float(pipe.predict_proba(row)[0, 1])
    return {
        "estimated_failure_probability": round(p_fail, 4),
        "flagged_for_inspection": bool(p_fail >= threshold),
        "decision_threshold_used": threshold,
        "note": "threshold is cost-optimized, not the default 0.5 — see /models/results.json for why",
    }


@app.get("/schedule")
def schedule(capacity_hours: float = WEEKLY_CAPACITY_HOURS):
    df = load_scored_fleet()
    scheduled, greedy_selected, candidates = solve_schedule(df, capacity_hours)
    return {
        "weekly_capacity_hours": capacity_hours,
        "machines_scheduled": int(len(scheduled)),
        "expected_dollar_value": round(float(scheduled["expected_net_benefit"].sum()), 2),
        "extra_value_vs_naive_risk_ranking": round(
            float(scheduled["expected_net_benefit"].sum() - greedy_selected["expected_net_benefit"].sum()), 2
        ),
        "schedule": scheduled[["Product ID", "Type", "p_fail", "hours", "expected_net_benefit"]]
        .round(4)
        .to_dict(orient="records"),
    }


@app.get("/explain/{product_id}")
def explain(product_id: str, use_llm: bool = False):
    try:
        return explain_machine(product_id, use_llm=use_llm)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/segments")
def segments(recompute: bool = False):
    """Per-segment (product-type) generalization analysis: does the single
    global threshold under- or over-perform on any segment, and is there
    enough data in that segment to trust a recalibrated one? See
    src/segment_analysis.py for the full caveats — L/M/H are product-quality
    variants used as a stand-in for deployment segments, not real customers."""
    path = RESULTS_DIR / "segment_results.json"
    if recompute or not path.exists():
        from src.segment_analysis import main as run_segment_analysis
        run_segment_analysis()
    return json.loads(path.read_text())


@app.get("/unsupervised")
def unsupervised(recompute: bool = False):
    """Clustering of actual failures (validated post-hoc against the known
    failure-mode flags) plus an IsolationForest anomaly detector checked
    as a safety net against failures the supervised model's threshold
    missed. See src/unsupervised_analysis.py for method and caveats."""
    path = RESULTS_DIR / "unsupervised_results.json"
    if recompute or not path.exists():
        from src.unsupervised_analysis import main as run_unsupervised_analysis
        run_unsupervised_analysis()
    return json.loads(path.read_text())
