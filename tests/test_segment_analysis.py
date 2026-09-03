from pathlib import Path

from src.segment_analysis import main as run_segment_analysis

ROOT = Path(__file__).resolve().parent.parent


def test_segment_analysis_flags_small_segments_and_is_non_negative():
    run_segment_analysis()
    import json
    results = json.loads((ROOT / "models" / "segment_results.json").read_text())

    assert set(results["segments"].keys()) == {"H", "L", "M"}

    # H has very few actual failures in the test slice -- must be flagged
    # as unreliable for per-segment recalibration, not silently recalibrated.
    assert results["segments"]["H"]["reliable_for_recalibration"] is False
    assert results["segments"]["H"]["n_actual_failures"] < 20

    # Segments with enough failures should be marked reliable
    assert results["segments"]["L"]["reliable_for_recalibration"] is True
    assert results["segments"]["M"]["reliable_for_recalibration"] is True

    # Recalibrating per segment should never cost more than the single
    # global threshold applied everywhere (it can only match or improve,
    # since the global threshold is always an available choice per segment).
    assert results["total_savings_from_segment_recalibration_usd"] >= 0

    # The framing note must explicitly disclaim "real customer sites"
    assert "not real customer sites" in results["framing"].lower() or \
           "not real customer" in results["framing"].lower()
