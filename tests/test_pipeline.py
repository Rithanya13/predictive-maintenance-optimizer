import json
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def test_dataset_shape_and_imbalance():
    df = pd.read_csv(ROOT / "data" / "ai4i2020.csv", encoding="utf-8-sig")
    assert df.shape[0] == 10000
    failure_rate = df["Machine failure"].mean()
    assert 0.01 < failure_rate < 0.10, "sanity check: this dataset should be rare-event imbalanced"


def test_model_artifact_predicts():
    bundle = joblib.load(ROOT / "models" / "failure_model.joblib")
    pipe = bundle["pipeline"]
    row = pd.DataFrame([{
        "Air temperature [K]": 298.1, "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551, "Torque [Nm]": 42.8, "Tool wear [min]": 0,
        "Type": "M",
    }])
    p = pipe.predict_proba(row)[0, 1]
    assert 0.0 <= p <= 1.0


def test_cost_optimal_threshold_beats_default_and_baselines():
    results = json.loads((ROOT / "models" / "results.json").read_text())
    default_cost = results["model_at_default_threshold_0.5"]["total_cost_usd"]
    optimal_cost = results["model_at_cost_optimal_threshold"]["total_cost"]
    run_to_failure = results["baseline_run_to_failure_usd"]
    inspect_everyone = results["baseline_inspect_everyone_usd"]

    assert optimal_cost <= default_cost, "cost-optimal threshold must not cost more than the naive default"
    assert default_cost < run_to_failure, "the model must beat doing nothing"
    assert default_cost < inspect_everyone, "the model must beat inspecting every machine"


def test_scheduler_respects_capacity_and_beats_naive_ranking():
    from src.schedule_optimizer import load_scored_fleet, solve_schedule

    df = load_scored_fleet()
    scheduled, greedy_selected, candidates = solve_schedule(df, capacity_hours=200)

    assert scheduled["hours"].sum() <= 200 + 1e-6
    assert scheduled["expected_net_benefit"].sum() >= greedy_selected["expected_net_benefit"].sum()


def test_explain_returns_grounded_output():
    from src.explain import explain_machine, load_scored_fleet_ids

    pid = load_scored_fleet_ids()[0]
    result = explain_machine(pid)
    assert result["product_id"] == pid
    assert 0.0 <= result["estimated_failure_probability"] <= 1.0
    assert len(result["top_drivers"]) > 0
    assert result["recommended_action"] in {
        "Inspect within 24-48 hours",
        "Schedule inspection this week",
        "Monitor; no immediate action required",
    }
