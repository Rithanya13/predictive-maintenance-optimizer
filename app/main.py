"""
FastAPI service exposing the three pieces as one product:
  POST /predict   -> failure risk for a machine's current sensor readings
  GET  /schedule  -> this week's optimized maintenance schedule
  GET  /explain/{product_id} -> plain-English reason a machine was flagged
  GET  /health
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.explain import explain_machine
from src.schedule_optimizer import load_scored_fleet, solve_schedule, WEEKLY_CAPACITY_HOURS
import joblib
from pathlib import Path as _Path

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
