"""Tests for HTTP endpoints."""

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from streaminfer.config import Settings
from streaminfer.server import create_app


@pytest.fixture
def app():
    settings = Settings(model_name="echo", batch_size=2, batch_timeout_ms=50)
    return create_app(settings)


class TestHTTPEndpoints:
    @pytest.mark.asyncio
    async def test_health(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/health")
            assert r.status_code == 200
            assert r.json()["status"] == "ok"
            assert r.json()["model"] == "echo"

    @pytest.mark.asyncio
    async def test_metrics_structure(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/metrics")
            assert r.status_code == 200
            data = r.json()
            assert "requests_total" in data
            assert "latency_p50_ms" in data
            assert "model_name" in data
            assert "batcher" in data

    @pytest.mark.asyncio
    async def test_reload_model(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.post("/api/reload", json={"model": "upper"})
            assert r.status_code == 200
            body = r.json()
            assert body["model"] == "upper"
            assert body["version"] == 1

    @pytest.mark.asyncio
    async def test_reload_unknown_model(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.post("/api/reload", json={"model": "nonexistent"})
            assert r.status_code == 500
            assert "error" in r.json()["status"]

    def test_predict_applies_http_backpressure_before_batching(self):
        settings = Settings(
            model_name="echo",
            batch_size=1,
            batch_timeout_ms=1,
            rate_limit_rps=0.1,
        )
        app = create_app(settings)

        with TestClient(app) as client:
            response = client.post("/predict", json={"text": "mlops"})

            assert response.status_code == 429
            assert response.json()["error"] == "rate limited"
            metrics = client.get("/metrics").json()
            assert metrics["requests_rejected"] == 1
            assert metrics["requests_total"] == 0
