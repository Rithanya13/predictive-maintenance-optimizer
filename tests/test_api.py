from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict():
    r = client.post("/predict", json={
        "air_temperature_k": 302.5,
        "process_temperature_k": 311.0,
        "rotational_speed_rpm": 1400,
        "torque_nm": 65.0,
        "tool_wear_min": 210,
        "type": "L",
    })
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["estimated_failure_probability"] <= 1.0
    assert "flagged_for_inspection" in body


def test_schedule():
    r = client.get("/schedule?capacity_hours=100")
    assert r.status_code == 200
    body = r.json()
    assert body["weekly_capacity_hours"] == 100
    assert isinstance(body["schedule"], list)


def test_explain_unknown_machine_404():
    r = client.get("/explain/NOT_A_REAL_ID")
    assert r.status_code == 404
