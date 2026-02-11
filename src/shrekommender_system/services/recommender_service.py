"""High level service that wires router and registry together."""

from __future__ import annotations

import logging
from typing import Optional

from shrekommender_system.config.center import ConfigCenter
from shrekommender_system.config.paths import PathsConfig
from shrekommender_system.config.serving import RouterConfig
from shrekommender_system.recommenders.base import RecommendationResult, SimilarItemsResult
from shrekommender_system.serving.context import RequestContext, build_context
from shrekommender_system.serving.model_registry import ModelRegistry
from shrekommender_system.serving.model_router import ModelRouter

logger = logging.getLogger(__name__)


class RecommenderService:
    def __init__(self, paths: Optional[PathsConfig] = None, *, poll_interval: float = 5.0):
        self.paths = paths or PathsConfig.from_env()
        self.paths.ensure()

        default_config = RouterConfig(default_model="als@v1")
        self.router = ModelRouter(default_config)
        self.registry = ModelRegistry(self.paths.models_root)
        self.config_center = ConfigCenter(self.paths.config_root, poll_interval=poll_interval)
        self.config_center.register("routes.yaml", self._on_router_config)
        self.config_center.start()

    def _on_router_config(self, payload: dict) -> None:
        logger.info("Applying router configuration update")
        self.router.apply_config(payload)

    # ------------------------------------------------------------------
    # Public API used by FastAPI layer

    def recommend(self, user_id: str, top_k: int = 10, context: Optional[RequestContext] = None) -> RecommendationResult:
        request_context = context or build_context(user_id)
        model_id = self.router.select(request_context)
        model = self.registry.ensure_loaded(model_id)
        try:
            result = model.recommend_for_user(user_id, top_k)
            return result
        except Exception as exc:
            logger.exception("Primary model %s failed: %s", model_id, exc)
            fallback_id = self.router.fallback()
            if fallback_id and fallback_id != model_id:
                logger.info("Falling back to %s", fallback_id)
                fallback_model = self.registry.ensure_loaded(fallback_id)
                return fallback_model.recommend_for_user(user_id, top_k)
            raise

    def similar_items(self, item_id: str, top_k: int = 10, context: Optional[RequestContext] = None) -> SimilarItemsResult:
        request_context = context or build_context("anonymous")
        model_id = self.router.select(request_context)
        model = self.registry.ensure_loaded(model_id)
        return model.get_similar_items(item_id, top_k)

    def get_user_profile(self, user_id: str, context: Optional[RequestContext] = None):
        request_context = context or build_context(user_id)
        model_id = self.router.select(request_context)
        model = self.registry.ensure_loaded(model_id)
        return model.get_user_profile(user_id)

    def health(self) -> dict:
        stats = {
            "models": {},
            "router": {
                "default_model": self.router.config.default_model,
                "rules": [rule.name for rule in self.router.config.rules],
            },
        }
        for model_id, model in self.registry.list_loaded().items():
            stats["models"][model_id] = model.health_check()
        return stats

    def apply_idle_policy(self) -> None:
        self.registry.apply_idle_policy()

    def reload_model(self, model_id: str) -> None:
        logger.info("Reloading model %s", model_id)
        self.registry.unload(model_id)
        self.registry.ensure_loaded(model_id)

    def unload_model(self, model_id: str) -> None:
        self.registry.unload(model_id)

    def shutdown(self) -> None:
        logger.info("Shutting down recommender service")
        self.config_center.stop()
        for model_id in list(self.registry.list_loaded().keys()):
            self.registry.unload(model_id)

    # ------------------------------------------------------------------
    # Introspection helpers

    def get_loaded_models(self) -> list[str]:
        return list(self.registry.list_loaded().keys())

    def get_router_overview(self) -> dict:
        config = self.router.config
        return {
            "default_model": config.default_model,
            "fallback_model": config.fallback_model or config.default_model,
            "rules": [
                {
                    "name": rule.name,
                    "model": rule.model,
                    "priority": rule.priority,
                    "match": rule.match,
                    "conditions": [
                        {
                            "field": cond.field,
                            "operator": cond.operator,
                            "value": cond.value,
                        }
                        for cond in rule.conditions
                    ],
                }
                for rule in config.rules
            ],
        }
