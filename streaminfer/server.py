"""FastAPI server with WebSocket streaming endpoint.

Two modes of operation:
  1. WebSocket at /ws — for streaming clients (send JSON, get JSON back)
  2. POST /predict — for simple request/response (batch internally)

Plus /metrics for monitoring and /api/reload for hot-swap.
"""

from __future__ import annotations

import json
import logging
import signal
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from .backpressure import ClientState
from .config import Settings
from .hotswap import ModelHolder, load_model
from .metrics import Metrics
from .pipeline import InferencePipeline

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    # shared state
    metrics = Metrics()
    initial_model = load_model(settings.model_name, settings.model_path)
    model_holder = ModelHolder(model=initial_model, name=settings.model_name)
    pipeline = InferencePipeline(
        model_holder=model_holder,
        metrics=metrics,
        batch_size=settings.batch_size,
        timeout_ms=settings.batch_timeout_ms,
    )

    # track connected clients for backpressure
    clients: dict[str, ClientState] = {}

    def _handle_sighup():
        """Reload model on SIGHUP."""
        try:
            new_model = load_model(settings.model_name, settings.model_path)
            model_holder.swap(new_model, settings.model_name)
        except Exception as e:
            logger.error("hot-swap failed: %s", e)

    @asynccontextmanager
    async def lifespan(app):
        await pipeline.start()

        # register SIGHUP for hot-swap (unix only)
        try:
            import asyncio

            loop = asyncio.get_running_loop()
            loop.add_signal_handler(signal.SIGHUP, _handle_sighup)
        except (NotImplementedError, OSError):
            pass  # windows or no SIGHUP support

        logger.info(
            "server started: model=%s, batch_size=%d",
            settings.model_name,
            settings.batch_size,
        )
        yield
        await pipeline.stop()
        logger.info("server stopped")

    app = FastAPI(title="StreamInfer", version="0.1.0", lifespan=lifespan)

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        client_id = f"{websocket.client.host}:{websocket.client.port}"
        clients[client_id] = ClientState(
            rate_limit=settings.rate_limit_rps,
            max_queue=settings.max_queue_size,
        )
        metrics.record_connect()
        client = clients[client_id]

        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    await websocket.send_json({"error": "invalid JSON"})
                    continue

                # backpressure check
                if not client.can_accept():
                    await websocket.send_json({
                        "error": "rate limited",
                        "retry_after_ms": int(client.bucket.wait_time() * 1000),
                    })
                    metrics.record_rejection()
                    continue

                if client.is_slow:
                    await websocket.send_json({"warning": "consumer falling behind"})

                client.on_request_start()
                try:
                    result = await pipeline.predict(data)
                    await websocket.send_json(result)
                finally:
                    client.on_request_done()

        except WebSocketDisconnect:
            pass
        finally:
            metrics.record_disconnect()
            clients.pop(client_id, None)

    @app.post("/predict")
    async def predict(data: dict):
        """Simple HTTP prediction endpoint (batched internally)."""
        result = await pipeline.predict(data)
        return result

    @app.get("/metrics")
    async def get_metrics():
        """Return current server metrics."""
        snapshot = metrics.snapshot()
        snapshot["model_name"] = model_holder.name
        snapshot["model_version"] = model_holder.version
        snapshot["batcher"] = {
            "total_batches": pipeline.batcher.total_batches,
            "total_items": pipeline.batcher.total_items,
            "total_timeouts": pipeline.batcher.total_timeouts,
        }
        return JSONResponse(snapshot)

    @app.post("/api/reload")
    async def reload_model(body: dict | None = None):
        """Hot-swap the model. Optionally pass {"model": "name", "path": "/path"}."""
        name = (body or {}).get("model", settings.model_name)
        path = (body or {}).get("path", settings.model_path)
        try:
            new_model = load_model(name, path)
            version = model_holder.swap(new_model, name)
            return {"status": "ok", "model": name, "version": version}
        except Exception as e:
            return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": model_holder.name}

    return app


def main():
    """Run the server directly."""
    import uvicorn

    settings = Settings()
    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
