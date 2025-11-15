"""Tests for HTTP endpoints.

The /predict endpoint goes through the batcher which needs an event loop,
so we skip it here and test prediction in test_pipeline.py instead.
Server endpoint tests focus on health, metrics, and reload.
"""

import pytest
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
