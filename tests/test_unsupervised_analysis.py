from src.unsupervised_analysis import main as run_unsupervised_analysis
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent


def test_unsupervised_analysis_runs_and_is_sane():
    run_unsupervised_analysis()
    results = json.loads((ROOT / "models" / "unsupervised_results.json").read_text())

    clustering = results["clustering"]
    assert 2 <= clustering["chosen_k"] <= 6
    assert -1.0 <= clustering["adjusted_rand_index_vs_known_failure_modes"] <= 1.0
    assert len(clustering["clusters"]) == clustering["chosen_k"]
    # every cluster's purity must be a valid proportion
    for c in clustering["clusters"]:
        assert 0.0 <= c["purity"] <= 1.0

    anomaly = results["anomaly_detection"]
    assert 0.0 <= anomaly["overall_anomaly_recall_of_actual_failures"] <= 1.0
    assert anomaly["supervised_model_missed_n_failures"] >= 0
    assert anomaly["of_those_missed_anomaly_detector_caught"] <= anomaly["supervised_model_missed_n_failures"]
