# Predictive Maintenance Decision Service

**The business problem, in plain terms:** a plant with thousands of machines
has two bad options. Do nothing and let machines fail — expensive emergency
repairs and unplanned downtime. Or inspect everything constantly — safe, but
wastes technician time and money on machines that were fine. Most predictive
maintenance projects stop at "here's a model with 97% accuracy," which
doesn't answer the question a plant manager actually has: *what should I do
this week, with the crew I actually have, and can I trust why you're telling
me to send someone to machine #4471?*

This project answers that question end to end: predict which machines are
actually at risk → convert that into a maintenance decision using real dollar
costs, not textbook accuracy → turn the flagged machines into an actual
weekly schedule under a limited technician-hour budget → explain every flag
in plain English so a non-technical floor manager will act on it.

## The results, in dollars (on held-out test data, not training data)

| Approach | Cost on test set |
|---|---|
| Do nothing, run machines to failure | $4,250,000 |
| Inspect every machine, every week | $5,510,000 |
| ML model, default 0.5 threshold (what most tutorials stop at) | $1,362,000 |
| **ML model, cost-optimized threshold (this project)** | **$1,218,000** |

Optimizing the decision threshold for dollars instead of accuracy — moving
it from 0.5 down to 0.21 — catches 8 more real failures (80 vs 72 out of 85)
by accepting more false alarms, because a missed failure costs 25x more than
a false alarm. That's **$144,000 saved over the "obvious" model**, and
**$3.03M saved (71%) versus doing no predictive maintenance at all**, on this
test set alone.

Separately, once machines are flagged, turning the flagged list into an
actual schedule with an optimizer (instead of just inspecting the top-N
riskiest machines) captures **an extra $475,000** in expected value, because
naive risk-ranking ignores that some machines take longer to inspect than
others — it fills the crew's calendar with the wrong mix of machines.

**Why the model beats both extremes:** doing nothing eats every failure at
full emergency-repair cost; inspecting everyone wastes money on the ~96% of
machines that were never going to fail. The model's job is finding the
narrow slice of machines actually worth a technician's time — and the
*threshold* decides how aggressively to lean toward catching failures versus
avoiding false alarms, which is a cost decision, not an ML metric.

## One global model does not perform identically everywhere

A single model deployed fleet-wide is usually evaluated with a single
global metric — which hides the fact that it can perform very differently
on different subgroups of the population it's deployed against. Using
AI4I's `Type` field (L/M/H — a **product-quality variant, not a real
customer site**) as a stand-in for "different deployment segments":

| Segment | Actual failures (test set) | Recall @ global threshold | FPR @ global threshold | Reliable to recalibrate? |
|---|---|---|---|---|
| H | 5 | 100% | 5.3% | **No — too few failures to trust a segment-specific threshold** |
| L | 51 | 94.1% | 7.3% | Yes |
| M | 29 | 93.1% | 6.3% → 3.1% at a recalibrated threshold of 0.45 | Yes |

Recalibrating the threshold per segment (only where there's enough data to
do it responsibly) saves an additional **$16,000** on top of the global
threshold's savings — a real but modest number, and reported as such rather
than inflated. The more important finding is qualitative: segment M's
optimal threshold (0.45) is more than double the global one (0.21), and
segment H simply doesn't have enough observed failures (5) in this dataset
to recalibrate against without overfitting to noise — which is itself the
finding a responsible analysis surfaces rather than hides. See
`src/segment_analysis.py` and `GET /segments`.

## Supervised isn't the whole story

Two unsupervised analyses, run with zero access to the failure label during
fitting:

- **Clustering the failures** (`src/unsupervised_analysis.py`, KMeans on raw
  sensor readings only) recovers a mechanically coherent sub-group: a
  37-machine cluster that is **84% pure "power failure" (PWF)**, characterized
  by unusually high rotational speed (2535 rpm vs. a ~1500 rpm population
  average) paired with unusually low torque (12.4 Nm) — physically sensible,
  since power failure in this dataset is driven by power (torque × speed)
  falling outside a valid band. Overall failure-mode recovery is weak
  (Adjusted Rand Index 0.04 against all five documented failure modes) —
  reported honestly rather than oversold, because most failure mechanisms
  don't separate cleanly on five raw sensor features alone.
- **Anomaly detection** (IsolationForest, contamination matched to the
  population failure rate) is tested as a safety net: of the failures the
  supervised model's cost-optimal threshold *missed*, did the unsupervised
  detector catch any of them anyway? On this test run the supervised
  threshold missed only 5 failures — too small a sample to draw a strong
  conclusion from, and the anomaly detector caught 0 of them, an honestly
  reported null result rather than a claimed win.

See `GET /unsupervised` for the full output.

## How it works

```
sensor readings ──► [1] failure model ──► [2] cost-optimal threshold ──► flagged machines
                                                                              │
                                                                              ▼
                                                          [3] scheduling optimizer
                                                          (fits flagged machines into
                                                           this week's crew-hour budget,
                                                           maximizing $ value protected)
                                                                              │
                                                                              ▼
                                                          [4] plain-English explanation
                                                          (SHAP-grounded, for a floor
                                                           manager — not a data scientist)
```

1. **`src/train_model.py`** — XGBoost classifier on the
   [AI4I 2020 Predictive Maintenance dataset](https://archive.ics.uci.edu/dataset/601)
   (10,000 real industrial machines, 3.39% failure rate — a realistic
   rare-event problem, not a balanced toy dataset). Sweeps the decision
   threshold and picks the one that **minimizes expected dollar cost**
   using configurable costs for a missed failure, a false alarm, and a
   caught failure, instead of defaulting to 0.5 or optimizing F1.
2. **`src/schedule_optimizer.py`** — an ILP (PuLP/CBC), not a sort. Given a
   weekly technician-hour budget and a flagged machine list (each with a
   failure probability and an inspection time that varies by machine type),
   it solves the 0/1 knapsack that maximizes expected dollars protected,
   and reports how much value a naive "just sort by risk score" approach
   would have left on the table.
3. **`src/explain.py`** — SHAP explains *why* a specific machine was
   flagged, and a template renders that as plain English ("torque is
   reading 67.5, higher than 90% of machines that have actually failed
   before"). An optional LLM pass (Claude, if `ANTHROPIC_API_KEY` is set)
   smooths the prose, strictly grounded in the same computed facts — no new
   claims. The project runs fully with zero API keys.
4. **`src/segment_analysis.py`** — same held-out test set, sliced by
   segment, to check whether the one global threshold is actually
   well-calibrated for every subgroup, and flags segments too small to
   recalibrate responsibly instead of pretending otherwise.
5. **`src/unsupervised_analysis.py`** — clusters the actual failures
   (validated post-hoc against documented failure modes, never trained on
   them) and runs an anomaly detector as an independent check on the
   supervised model's blind spots.
6. **`app/main.py`** — a FastAPI service wrapping all of the above:
   `POST /predict`, `GET /schedule`, `GET /explain/{product_id}`,
   `GET /segments`, `GET /unsupervised`.

## Why the numbers are honest, not cherry-picked

- All dollar figures above are computed on a **held-out 25% test set**, not
  training data.
- `TWF/HDF/PWF/OSF/RNF` (failure-mode flags in the raw dataset) are
  **excluded from features** — they're only known after a failure happens,
  so including them would leak the label.
- The $50k / $2k / $8k cost assumptions (missed failure / false alarm /
  caught failure) are clearly labeled as configurable business inputs in
  `src/train_model.py` — a real deployment pulls these from the customer's
  finance/maintenance team, not from a data scientist's guess. Change them
  and the optimal threshold moves accordingly — that's the point.
- The scheduling "hours per machine type" is an explicit, documented
  stand-in (AI4I's `Type` field is a product-quality variant, not an
  inspection-complexity label) since the public dataset has no real
  work-order data — flagged clearly in the code rather than presented as
  real customer data.

## Running it

```bash
pip install -r requirements.txt
python3 src/train_model.py            # trains model, writes models/results.json
python3 src/schedule_optimizer.py     # builds this week's schedule
python3 src/explain.py                # demo explanations for the 3 riskiest machines
python3 src/segment_analysis.py       # per-segment generalization check
python3 src/unsupervised_analysis.py  # clustering + anomaly detection
uvicorn app.main:app --reload         # API at http://localhost:8000/docs
```

Or with Docker:

```bash
docker build -t predictive-maintenance-optimizer .
docker run -p 8000:8000 predictive-maintenance-optimizer
```

Tests (13 passing, covering the model, the optimizer's capacity constraint,
the segment analysis, the unsupervised analysis, and the API):

```bash
pytest tests/ -v
```

CI (`.github/workflows/ci.yml`) retrains the model from scratch on every
push and runs the full test suite plus a Docker build, so the numbers above
are reproducible on a clean checkout, not just on one machine.

## What this maps to in an enterprise AI platform context

This is the actual shape of enterprise predictive-maintenance products
(fleet-wide monitoring, cost-sensitive alerting, prescriptive scheduling,
plain-language explanations for non-technical operators, and awareness that
one global model doesn't behave identically everywhere it's deployed) — not
a Kaggle leaderboard exercise. Deliberately **not** built: distributed
training/serving infrastructure (Spark/Ray) for a 10,000-row dataset would
be manufactured complexity, not a real need — the Docker/FastAPI/CI path is
the credible production story at this scale. The open items a real
deployment would still add: model monitoring for data drift over time, and
replacing the static hours-per-machine-type assumption with real work-order
data.
