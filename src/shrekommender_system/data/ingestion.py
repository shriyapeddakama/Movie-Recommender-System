"""Simple data ingestion from Kafka"""

import json
import subprocess
from datetime import datetime
import logging

from shrekommender_system.config.data import KafkaConfig, IngestionConfig
from shrekommender_system.config.paths import PathsConfig
from shrekommender_system.data.schema import (
    parse_event,
    WatchEvent,
    RateEvent,
    RecommendationEvent,
)

logger = logging.getLogger(__name__)


class KafkaDataIngester:
    """Simple Kafka data ingester that separates events by type"""

    def __init__(self):
        self.kafka_config = KafkaConfig(consumer_group="shrekommender-ingester")
        self.paths = PathsConfig.from_env()
        self.paths.ensure()
        self.paths.ensure_data_dirs("raw", "processed", "intermediate")
        self.ingestion_config = IngestionConfig()

    def fetch_recent(self, n: int = 100000) -> dict:
        """Fetch the most recent N records from Kafka and save by event type and date

        Args:
            n: Number of recent records to fetch

        Returns:
            Statistics dictionary
        """
        logger.info(f"Fetching last {n} records from Kafka")

        # First, get the last offset to calculate where to start
        get_end_cmd = [
            "kcat", "-C",
            "-b", self.kafka_config.bootstrap_servers,
            "-t", self.kafka_config.topic,
            "-o", "end",
            "-c", "1",
            "-f", "%o",
            "-q"
        ]

        result = subprocess.run(get_end_cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0 or not result.stdout.strip():
            logger.error("Failed to get topic size")
            return {"status": "failed", "error": "Could not determine topic size"}

        last_offset = int(result.stdout.strip())
        start_offset = max(0, last_offset - n)

        logger.info(f"Topic has {last_offset} messages, fetching from offset {start_offset}")

        # Fetch N records from calculated offset
        cmd = [
            "kcat", "-C",
            "-b", self.kafka_config.bootstrap_servers,
            "-t", self.kafka_config.topic,
            "-o", str(start_offset),
            "-c", str(n),
            "-q"
        ]

        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            logger.error(f"Failed to fetch data: {result.stderr}")
            return {"status": "failed", "error": result.stderr}

        # Parse and group events by type
        events_by_type = {
            "watch": [],
            "rate": [],
            "recommendation": []
        }

        stats = {
            "status": "completed",
            "total_events": 0,
            "watch_events": 0,
            "rate_events": 0,
            "recommendation_events": 0,
            "parse_errors": 0,
            "files_created": []
        }

        for line in result.stdout.strip().split('\n'):
            if not line:
                continue

            event = parse_event(line)
            if event:
                # Group by event type
                if isinstance(event, WatchEvent):
                    events_by_type["watch"].append(event)
                    stats["watch_events"] += 1
                elif isinstance(event, RateEvent):
                    events_by_type["rate"].append(event)
                    stats["rate_events"] += 1
                elif isinstance(event, RecommendationEvent):
                    events_by_type["recommendation"].append(event)
                    stats["recommendation_events"] += 1
                stats["total_events"] += 1
            else:
                stats["parse_errors"] += 1

        # Generate timestamp for file names
        ingest_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save events to separate files by type with timestamp and count in filename
        for event_type, events in events_by_type.items():
            if not events:
                continue

            # Create filename with timestamp and count
            file_name = f"{event_type}_{ingest_time}_{len(events)}.jsonl"
            file_path = self.paths.data_path("raw", file_name)

            # Save to file
            with open(file_path, 'w') as f:
                for event in events:
                    f.write(json.dumps(event.to_dict()) + '\n')

            logger.info(f"Saved {len(events)} {event_type} events to {file_path}")
            stats["files_created"].append(str(file_path))

        return stats


def ingest_recent(n: int = 100000) -> dict:
    """Main entry point to fetch recent data

    Args:
        n: Number of recent records to fetch

    Returns:
        Ingestion statistics
    """
    ingester = KafkaDataIngester()
    return ingester.fetch_recent(n)
