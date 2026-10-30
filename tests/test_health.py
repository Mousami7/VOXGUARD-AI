from fastapi.testclient import TestClient


def test_root_returns_running_status(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"name": "VoxGuard AI", "status": "running"}
    assert "X-Request-ID" in response.headers


def test_health_returns_healthy(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
