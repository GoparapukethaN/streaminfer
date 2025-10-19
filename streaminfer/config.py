"""Server configuration.

Uses pydantic-settings so you can override anything via environment variables.
Prefix: STREAMINFER_  (e.g. STREAMINFER_BATCH_SIZE=32)
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "STREAMINFER_"}

    host: str = "0.0.0.0"
    port: int = 8000

    # batching
    batch_size: int = 16
    batch_timeout_ms: int = 50  # flush after this many ms even if batch isn't full

    # backpressure
    max_queue_size: int = 1000  # per-client queue limit
    rate_limit_rps: float = 100.0  # requests per second per client

    # model
    model_name: str = "echo"  # default to echo model for testing
    model_path: str | None = None

    # hot swap
    config_path: str | None = None  # path to model config JSON for hot-swap
