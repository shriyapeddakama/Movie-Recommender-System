"""Placeholder training pipeline for lifecycle automation.

This module provides the scaffolding that future training jobs can plug into.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from shrekommender_system.config.paths import PathsConfig
from shrekommender_system.config.training import TrainingSpec

logger = logging.getLogger(__name__)


@dataclass
class TrainingJobConfig:
    """Describes a training task pulled from ``configs/training``."""

    model_id: str
    training_config: Dict[str, Any] = field(default_factory=dict)
    output_override: Optional[Path] = None


def run_training(job: TrainingJobConfig, paths: Optional[PathsConfig] = None) -> Path:
    """Execute the training job.

    Currently this is a placeholder that only sets up filesystem structure and
    emits logging so callers can integrate scheduling and orchestration without
    the actual training implementation being ready.

    Returns the path that would contain the trained model artefacts.
    """

    paths = paths or PathsConfig.from_env()
    paths.ensure()

    output_root = job.output_override or paths.models_root / job.model_id
    output_root.mkdir(parents=True, exist_ok=True)

    hyperparameters = job.training_config.get("hyperparameters", {})
    data_sources = job.training_config.get("inputs", {})
    runtime = job.training_config.get("runtime", {})

    logger.info(
        "Planning training job for %s (output=%s, hyperparameters=%s)",
        job.model_id,
        output_root,
        hyperparameters,
    )
    if data_sources:
        logger.info("Data sources: %s", data_sources)
    if runtime:
        logger.info("Runtime options: %s", runtime)
    # Placeholder for actual training logic.
    # Integrations should replace this section with data loading, model fitting,
    # metric calculation, and artefact materialisation.
    (output_root / "TRAINING_PENDING").touch()

    logger.info("Training placeholder completed for %s", job.model_id)
    return output_root


def job_from_spec(spec: TrainingSpec) -> TrainingJobConfig:
    """Convert a :class:`TrainingSpec` into executable job configuration."""

    output_override = spec.output_dir()
    return TrainingJobConfig(
        model_id=spec.model_id,
        training_config=spec.to_payload(),
        output_override=output_override,
    )
