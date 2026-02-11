"""Training configuration schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from .loader import load_yaml


@dataclass
class TrainingSpec:
    """Represents the declarative configuration for a training job."""

    model_id: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    runtime: Dict[str, Any] = field(default_factory=dict)
    publish: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "TrainingSpec":
        if "model_id" not in payload:
            raise ValueError("Training spec requires 'model_id'")
        return TrainingSpec(
            model_id=payload["model_id"],
            inputs=payload.get("inputs", {}),
            hyperparameters=payload.get("hyperparameters", {}),
            runtime=payload.get("runtime", {}),
            publish=payload.get("publish", {}),
            metadata=payload.get("metadata", {}),
        )

    def output_dir(self) -> Optional[Path]:
        location = self.publish.get("output_dir")
        return Path(location) if location else None

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model_id": self.model_id,
            "inputs": self.inputs,
            "hyperparameters": self.hyperparameters,
            "runtime": self.runtime,
            "publish": self.publish,
            "metadata": self.metadata,
        }
        return payload


def load_training_spec(path: Path) -> TrainingSpec:
    return TrainingSpec.from_dict(load_yaml(path))
