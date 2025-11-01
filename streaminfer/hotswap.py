"""Atomic model hot-swap.

The idea is simple: load the new model in a background thread, then swap
the pointer under a lock. In-flight requests finish with the old model,
new requests get the new one. No downtime.

Trigger a swap by:
  1. Sending SIGHUP to the process
  2. POST to /api/reload
  3. Dropping a new config file (if using config_path)
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)


class ModelHolder:
    """Thread-safe holder for the current model.

    Swap is atomic from the caller's perspective — reads never block
    on writes because we swap a single pointer under a lock.
    """

    def __init__(self, model: Any = None, name: str = "default"):
        self._model = model
        self._name = name
        self._lock = threading.Lock()
        self._version = 0

    @property
    def model(self) -> Any:
        with self._lock:
            return self._model

    @property
    def name(self) -> str:
        with self._lock:
            return self._name

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    def swap(self, new_model: Any, new_name: str | None = None) -> int:
        """Atomically replace the current model.

        Returns the new version number.
        """
        with self._lock:
            old_name = self._name
            self._model = new_model
            if new_name:
                self._name = new_name
            self._version += 1
            version = self._version

        logger.info(
            "model swapped: %s -> %s (version %d)",
            old_name,
            new_name or old_name,
            version,
        )
        return version


def load_model(name: str, path: str | None = None) -> Any:
    """Load a model by name.

    Built-in models:
      - "echo": returns input as-is (for testing)
      - "upper": uppercases text input (for testing)

    For real models, provide a path to a .pt/.pkl file.
    """
    if name == "echo":
        return EchoModel()
    elif name == "upper":
        return UpperModel()
    elif path:
        return _load_from_file(path)
    else:
        raise ValueError(f"unknown model: {name} (no path provided)")


def _load_from_file(path: str) -> Any:
    """Load a serialized model from disk."""
    if path.endswith((".pt", ".pth")):
        import torch

        return torch.load(path, map_location="cpu", weights_only=False)
    elif path.endswith((".pkl", ".joblib")):
        import joblib

        return joblib.load(path)
    else:
        raise ValueError(f"unsupported model format: {path}")


class EchoModel:
    """Test model that returns the input unchanged."""

    def predict(self, inputs: list[dict]) -> list[dict]:
        return [{"result": inp.get("text", ""), "model": "echo"} for inp in inputs]


class UpperModel:
    """Test model that uppercases text."""

    def predict(self, inputs: list[dict]) -> list[dict]:
        return [
            {"result": inp.get("text", "").upper(), "model": "upper"} for inp in inputs
        ]
