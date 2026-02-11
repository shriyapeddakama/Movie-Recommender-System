"""Filesystem layout helpers based on environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

@dataclass
class PathsConfig:
    config_root: Path
    models_root: Path
    data_root: Path

    @classmethod
    def from_env(cls) -> "PathsConfig":
        base_dir = Path(os.getenv("SHREK_BASE_DIR", ".")).resolve()
        config_root = Path(os.getenv("SHREK_CONFIG_DIR", base_dir / "configs")).resolve()
        models_root = Path(os.getenv("SHREK_MODELS_DIR", base_dir / "models")).resolve()
        data_root = Path(os.getenv("SHREK_DATA_DIR", base_dir / "data")).resolve()
        return cls(config_root=config_root, models_root=models_root, data_root=data_root)

    def ensure(self) -> None:
        for path in (self.config_root, self.models_root, self.data_root):
            path.mkdir(parents=True, exist_ok=True)

    def ensure_data_dirs(self, *subdirs: str) -> None:
        if not subdirs:
            subdirs = ("raw", "processed", "intermediate")
        for name in subdirs:
            (self.data_root / name).mkdir(parents=True, exist_ok=True)

    def config_path(self, relative: str) -> Path:
        return self.config_root / relative

    def model_path(self, identifier: str) -> Path:
        return self.models_root / identifier

    def data_path(self, *relative: str) -> Path:
        return self.data_root.joinpath(*relative)
