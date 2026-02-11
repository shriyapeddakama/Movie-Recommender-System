"""Model registry responsible for loading model instances on demand."""

from __future__ import annotations

import importlib
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Type

from shrekommender_system.config.loader import load_yaml
from shrekommender_system.config.serving import ModelManifest
from shrekommender_system.recommenders.base import BaseRecommender

logger = logging.getLogger(__name__)


@dataclass
class ModelHandle:
    definition: ModelManifest
    instance: BaseRecommender
    last_access: float


class ModelRegistry:
    """Lazy loading registry with basic idle GC support."""

    def __init__(self, models_root: Path, max_idle_seconds: int = 900):
        self.models_root = models_root
        self.max_idle_seconds = max_idle_seconds
        self._loaded: Dict[str, ModelHandle] = {}
        self._lock = threading.RLock()

    def apply_idle_policy(self) -> None:
        now = time.time()
        with self._lock:
            to_unload = [
                model_id
                for model_id, handle in self._loaded.items()
                if now - handle.last_access > self.max_idle_seconds
            ]
        for model_id in to_unload:
            logger.info("Unloading idle model %s", model_id)
            self.unload(model_id)

    def ensure_loaded(self, model_id: str) -> BaseRecommender:
        with self._lock:
            handle = self._loaded.get(model_id)
            if handle:
                handle.last_access = time.time()
                return handle.instance

            definition = self._load_definition(model_id)
            model_class = self._resolve_class(definition.class_path)
            model_dir = self.models_root / model_id
            runtime_config = dict(definition.runtime)
            runtime_config.setdefault("artefacts", definition.artefacts)
            runtime_config.setdefault("metadata", definition.metadata)

            instance = model_class(runtime_config, model_name=definition.identifier)
            instance.load_model(str(model_dir))
            instance.warmup()

            handle = ModelHandle(
                definition=definition,
                instance=instance,
                last_access=time.time(),
            )
            self._loaded[model_id] = handle
            logger.info("Loaded model %s from %s", model_id, model_dir)
            return instance

    def unload(self, model_id: str) -> None:
        with self._lock:
            handle = self._loaded.pop(model_id, None)
        if handle:
            try:
                handle.instance.teardown()
            finally:
                logger.info("Model %s unloaded", model_id)

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            return {
                model_id: {
                    "last_access": handle.last_access,
                }
                for model_id, handle in self._loaded.items()
            }

    def list_loaded(self) -> Dict[str, BaseRecommender]:
        with self._lock:
            return {model_id: handle.instance for model_id, handle in self._loaded.items()}

    def _load_definition(self, model_id: str) -> ModelManifest:
        model_dir = self.models_root / model_id
        config_path = model_dir / "manifest.yaml"
        payload = load_yaml(config_path)
        definition = ModelManifest.from_dict(payload)
        actual_id = definition.identifier
        if actual_id != model_id:
            logger.warning(
                "Model directory %s declares identifier %s", model_id, actual_id
            )
        return definition

    @staticmethod
    def _resolve_class(path: str) -> Type[BaseRecommender]:
        module_name, _, class_name = path.partition(":")
        if not module_name or not class_name:
            raise ValueError(f"Invalid class path: {path}")
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        if not issubclass(cls, BaseRecommender):  # type: ignore[arg-type]
            raise TypeError(f"{path} is not a BaseRecommender subclass")
        return cls  # type: ignore[return-value]
