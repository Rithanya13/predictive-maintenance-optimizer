import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest


@pytest.fixture(scope="session", autouse=True)
def ensure_model_trained():
    """CI runs on a clean checkout with no model artifact, so train once
    per test session if it isn't already there."""
    model_path = ROOT / "models" / "failure_model.joblib"
    if not model_path.exists():
        from src.train_model import main
        main()
    yield
