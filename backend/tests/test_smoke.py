import os

os.environ.setdefault("DB_PASSWORD", "test-password")
os.environ.setdefault("TEST_MODE", "1")
os.environ.setdefault("LOG_LEVEL", "CRITICAL")
os.environ.setdefault("LOG_FORMAT", "json")

from fastapi.testclient import TestClient

from backend.main import app, _TEST_STUB_CHUNKS  # type: ignore


def _make_client() -> TestClient:
    return TestClient(app)


def test_health_endpoint_returns_ok():
    with _make_client() as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_query_endpoint_uses_stubbed_response():
    payload = {"query": "How many stubs?", "user_groups": ["stub"]}

    with _make_client() as client:
        response = client.post("/query", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["response"].startswith("Stub response generated during test mode")
    assert data["sources"], "Expected at least one stub source"
    assert data["sources"][0]["doc_id"] == _TEST_STUB_CHUNKS[0]["doc_id"]


def test_langchain_endpoint_short_circuits_in_test_mode():
    payload = {"query": "LangChain stub?", "user_groups": ["stub"]}

    with _make_client() as client:
        response = client.post("/query-lc", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["response"].startswith("Stub LangChain response")
    assert data["sources"], "Expected stub sources in test response"

