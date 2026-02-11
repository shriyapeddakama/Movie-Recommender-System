"""Shared configuration dataclasses used by multiple domains."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass
class KafkaConfig:
    bootstrap_servers: str = field(default_factory=lambda: _env("SHREK_KAFKA_BOOTSTRAP", "localhost:9092"))
    topic: str = field(default_factory=lambda: _env("SHREK_KAFKA_TOPIC", "movielog11"))
    consumer_group: str = "shrekommender-default"
    fetch_timeout_ms: int = 10000
    max_poll_records: int = 10000


@dataclass
class IngestionConfig:
    batch_size: int = 1000
    skip_existing: bool = True
    verify_data: bool = True
