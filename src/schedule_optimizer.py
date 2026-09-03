"""
Turn "risk scores" into an actual weekly maintenance schedule.

A risk score alone is not a decision. A plant has a fixed number of
technician-hours per week and cannot inspect every flagged machine, so
someone has to decide WHICH flagged machines get a technician this week.

This is a classic 0/1 knapsack: each machine has
  - an expected dollar benefit if inspected this week, and
  - a technician-hour cost to inspect it (varies by machine complexity),
and the crew has a fixed weekly hour budget.

We solve it exactly with an ILP (PuLP/CBC) instead of just sorting by risk
score, because sorting by risk score alone ignores that inspection time
differs by machine type -- a purely risk-ranked list can leave real dollars
on the table versus the value-per-hour-optimal selection.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pulp

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "failure_model.joblib"
DATA_PATH = ROOT / "data" / "ai4i2020.csv"
SCHEDULE_PATH = ROOT / "models" / "weekly_schedule.json"

# Cost assumptions (kept in sync with train_model.py)
COST_FALSE_NEGATIVE = 50_000
COST_FALSE_POSITIVE = 2_000
COST_TRUE_POSITIVE = 8_000

# Illustrative inspection-time-by-machine-type assumption (a real deployment
# would pull this from the customer's CMMS/work-order system). AI4I's
# "Type" field (L/M/H) is a product quality variant; we use it here as a
# stand-in for "inspection complexity" to make the capacity constraint concrete.
HOURS_BY_TYPE = {"L": 1.5, "M": 2.0, "H": 3.0}

WEEKLY_CAPACITY_HOURS = 200  # e.g. 5 technicians x 40 hours/week


def expected_net_benefit(p_fail: float) -> float:
    """Expected $ benefit of inspecting a machine with failure probability p_fail,
    vs. leaving it alone. Positive = worth inspecting in isolation (ignoring capacity)."""
    benefit_if_failure_caught = p_fail * (COST_FALSE_NEGATIVE - COST_TRUE_POSITIVE)
    cost_if_actually_healthy = (1 - p_fail) * COST_FALSE_POSITIVE
    return benefit_if_failure_caught - cost_if_actually_healthy


def load_scored_fleet():
    bundle = joblib.load(MODEL_PATH)
    pipe = bundle["pipeline"]
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    X = df[bundle["features_numeric"] + bundle["features_categorical"]]
    df = df.copy()
    df["p_fail"] = pipe.predict_proba(X)[:, 1]
    df["hours"] = df["Type"].map(HOURS_BY_TYPE)
    df["expected_net_benefit"] = df["p_fail"].apply(expected_net_benefit)
    return df


def solve_schedule(df: pd.DataFrame, capacity_hours: float):
    # Only worth considering machines where inspecting has positive expected
    # value in isolation -- no point letting the ILP burn time on obviously-fine machines.
    candidates = df[df["expected_net_benefit"] > 0].reset_index(drop=True)

    prob = pulp.LpProblem("weekly_maintenance_schedule", pulp.LpMaximize)
    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in candidates.index}

    prob += pulp.lpSum(candidates.loc[i, "expected_net_benefit"] * x[i] for i in candidates.index)
    prob += pulp.lpSum(candidates.loc[i, "hours"] * x[i] for i in candidates.index) <= capacity_hours

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    candidates["scheduled"] = [int(pulp.value(x[i])) for i in candidates.index]
    scheduled = candidates[candidates["scheduled"] == 1].sort_values("expected_net_benefit", ascending=False)

    # Naive baseline: greedy by risk score alone (ignoring hours-per-machine), same capacity
    greedy = candidates.sort_values("p_fail", ascending=False).copy()
    greedy["cum_hours"] = greedy["hours"].cumsum()
    greedy_selected = greedy[greedy["cum_hours"] <= capacity_hours]

    return scheduled, greedy_selected, candidates


def main():
    df = load_scored_fleet()
    scheduled, greedy_selected, candidates = solve_schedule(df, WEEKLY_CAPACITY_HOURS)

    optimizer_value = scheduled["expected_net_benefit"].sum()
    greedy_value = greedy_selected["expected_net_benefit"].sum()

    summary = {
        "weekly_capacity_hours": WEEKLY_CAPACITY_HOURS,
        "machines_worth_considering": int(len(candidates)),
        "machines_scheduled_by_optimizer": int(len(scheduled)),
        "hours_used_by_optimizer": float(scheduled["hours"].sum()),
        "expected_dollar_value_optimizer": round(float(optimizer_value), 2),
        "machines_scheduled_by_naive_risk_ranking": int(len(greedy_selected)),
        "hours_used_by_naive_ranking": float(greedy_selected["hours"].sum()),
        "expected_dollar_value_naive_ranking": round(float(greedy_value), 2),
        "extra_value_captured_by_optimizer_usd": round(float(optimizer_value - greedy_value), 2),
        "top_10_scheduled_machines": scheduled[
            ["Product ID", "Type", "p_fail", "hours", "expected_net_benefit"]
        ].head(10).round(4).to_dict(orient="records"),
    }

    SCHEDULE_PATH.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
