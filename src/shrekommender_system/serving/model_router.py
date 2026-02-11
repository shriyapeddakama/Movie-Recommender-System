"""Routing logic that maps a request to a model identifier."""

from __future__ import annotations

import logging
from typing import Optional

from shrekommender_system.config.serving import RouterConfig
from .context import RequestContext

logger = logging.getLogger(__name__)


class ModelRouter:
    def __init__(self, config: RouterConfig):
        self._config = config

    @property
    def config(self) -> RouterConfig:
        return self._config

    def apply_config(self, payload: dict) -> None:
        self._config = RouterConfig.from_dict(payload)
        logger.info(
            "Router config updated: default=%s, rules=%s",
            self._config.default_model,
            [rule.name for rule in self._config.rules],
        )

    def select(self, context: RequestContext) -> str:
        if context.model_hint:
            logger.debug("Using model hint from request: %s", context.model_hint)
            return context.model_hint

        resolver = context.get
        for rule in self._config.rules:
            if rule.matches(resolver):
                logger.debug("Rule '%s' matched model '%s'", rule.name, rule.model)
                return rule.model

        logger.debug("No router rules matched, using default model '%s'", self._config.default_model)
        return self._config.default_model

    def fallback(self) -> Optional[str]:
        return self._config.fallback_model or self._config.default_model
