#!/usr/bin/env python3
"""Standalone Kafka ingestion script for the Shrekommender project.

The script reads new records from the configured Kafka topic, splits them by
event type (watch/rate/recommendation), and saves each type into its own JSONL
file under the chosen output directory.

By default the script reads at most the latest N events (configurable) without
committing offsets, so each run operates on the tail of the topic regardless of
prior executions.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

from kafka import KafkaConsumer  # type: ignore


def _default_env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simplified Kafka ingestion for the Shrekommender system."
    )
    parser.add_argument(
        "--bootstrap",
        default=_default_env("SHREK_KAFKA_BOOTSTRAP", "localhost:9092"),
        help="Kafka bootstrap servers (default env SHREK_KAFKA_BOOTSTRAP or %(default)s)",
    )
    parser.add_argument(
        "--topic",
        default=_default_env("SHREK_KAFKA_TOPIC", "movielog11"),
        help="Kafka topic to consume (default env SHREK_KAFKA_TOPIC or %(default)s)",
    )
    parser.add_argument(
        "--group",
        default=_default_env("SHREK_KAFKA_GROUP", ""),
        help="Consumer group identifier (default env SHREK_KAFKA_GROUP or empty for none)",
    )
    parser.add_argument(
        "--output-dir",
        default=_default_env("RAW_DATA_DIR", "data/raw"),
        help="Directory for resulting JSONL files (default env RAW_DATA_DIR or %(default)s)",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=int(_default_env("SHREK_MAX_EVENTS", "1000")),
        help="Maximum number of newest events to ingest (default env SHREK_MAX_EVENTS or %(default)s)",
    )
    return parser.parse_args()


def parse_event(line: str) -> Optional[Dict[str, object]]:
    """Parse a single log line using the same rules as the original schema module."""
    try:
        parts = line.strip().split(",", 2)
        if len(parts) < 3:
            return None

        timestamp = parts[0]
        user_id = parts[1]
        action = parts[2]

        if action.startswith("GET /data/m/"):
            match = re.match(r"GET /data/m/([^/]+)/(\d+)\.mpg", action)
            if not match:
                return None
            return {
                "timestamp": timestamp,
                "user_id": user_id,
                "event_type": "watch",
                "movie_id": match.group(1),
                "minute": int(match.group(2)),
            }

        if action.startswith("GET /rate/"):
            match = re.match(r"GET /rate/([^=]+)=(\d+)", action)
            if not match:
                return None
            return {
                "timestamp": timestamp,
                "user_id": user_id,
                "event_type": "rate",
                "movie_id": match.group(1),
                "rating": int(match.group(2)),
            }

        if "recommendation request" in action:
            full_line = line.strip()
            server_match = re.search(r"recommendation request ([^,]+)", full_line)
            status_match = re.search(r"status (\d+)", full_line)
            result_match = re.search(r"result: (\[.*\])", full_line)
            parts = full_line.split(",")
            response_time = parts[-1] if len(parts) > 5 else None

            if not server_match or not status_match:
                return None
            return {
                "timestamp": timestamp,
                "user_id": user_id,
                "event_type": "recommendation",
                "server": server_match.group(1).strip(),
                "status_code": int(status_match.group(1)),
                "recommendations": result_match.group(1).strip() if result_match else "",
                "response_time": response_time,
            }

        return None
    except Exception:
        return None


def write_events(events: Dict[str, List[Dict[str, object]]], output_dir: Path) -> List[Path]:
    created_files: List[Path] = []

    for event_type, payload in events.items():
        if not payload:
            continue
        file_path = output_dir / f"{event_type}.jsonl"
        with file_path.open("w", encoding="utf-8") as handle:
            for entry in payload:
                json.dump(entry, handle)
                handle.write("\n")
        created_files.append(file_path)

    return created_files


def write_metadata(metadata: Dict[str, object], output_dir: Path) -> Path:
    path = output_dir / "metadata.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    return path


def _ensure_assignment(consumer: KafkaConsumer, timeout: float = 10.0) -> Set:
    """Wait until Kafka assigns partitions to this consumer."""
    end_time = time.time() + timeout
    assignment = consumer.assignment()
    while not assignment:
        consumer.poll(timeout_ms=100)
        assignment = consumer.assignment()
        if assignment:
            break
        if time.time() >= end_time:
            raise RuntimeError("Kafka did not assign any partitions within timeout.")
    return assignment


def _tp_map_to_serializable(mapping: Dict) -> Dict[str, int]:
    return {f"{tp.topic}:{tp.partition}": offset for tp, offset in mapping.items()}


def consume_once(args: argparse.Namespace) -> Dict[str, object]:
    consumer = KafkaConsumer(
        args.topic,
        bootstrap_servers=args.bootstrap,
        group_id=args.group or None,
        auto_offset_reset="latest",
        enable_auto_commit=False,
        value_deserializer=lambda value: value.decode("utf-8"),
    )

    assignment = _ensure_assignment(consumer)
    target_offsets = consumer.end_offsets(assignment)
    beginning_offsets = consumer.beginning_offsets(assignment)

    per_partition_window = None
    if args.max_events > 0 and assignment:
        per_partition_window = max(1, (args.max_events + len(assignment) - 1) // len(assignment))

    start_positions = {}
    processed_offsets = {}
    for tp in assignment:
        if per_partition_window is None:
            start_offset = beginning_offsets[tp]
        else:
            start_offset = max(beginning_offsets[tp], target_offsets[tp] - per_partition_window)
        consumer.seek(tp, start_offset)
        start_positions[tp] = start_offset
        processed_offsets[tp] = start_offset - 1

    events: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    stats = {
        "total_events": 0,
        "watch": 0,
        "rate": 0,
        "recommendation": 0,
        "parse_errors": 0,
    }

    stats["start_offsets"] = _tp_map_to_serializable(start_positions)
    stats["target_offsets"] = _tp_map_to_serializable(target_offsets)

    needs_work = any(start_positions[tp] < target_offsets[tp] for tp in assignment)

    try:
        while needs_work:
            records = consumer.poll(timeout_ms=1000)
            if not records:
                continue

            for tp, msgs in records.items():
                for msg in msgs:
                    if msg.offset >= target_offsets[tp]:
                        continue

                    parsed = parse_event(msg.value)
                    if not parsed:
                        stats["parse_errors"] += 1
                        continue

                    events[parsed["event_type"]].append(parsed)
                    stats["total_events"] += 1
                    stats[parsed["event_type"]] += 1
                    processed_offsets[tp] = msg.offset

                    if args.max_events > 0 and stats["total_events"] >= args.max_events:
                        needs_work = False
                        break
                if not needs_work:
                    break

            needs_work = any(
                processed_offsets[tp] < target_offsets[tp] - 1 for tp in assignment
            )
            if args.max_events > 0 and stats["total_events"] >= args.max_events:
                break
    finally:
        consumer.close()

    stats["end_offsets"] = _tp_map_to_serializable(processed_offsets)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats["files_created"] = [str(path) for path in write_events(events, output_dir)]
    stats["run_timestamp"] = datetime.now().isoformat()
    metadata_path = write_metadata(stats, output_dir)
    stats["metadata_file"] = str(metadata_path)
    return stats


def main() -> None:
    args = parse_args()
    summary = consume_once(args)

    if summary["files_created"]:
        print("Ingestion completed.")
        print(f"Total events: {summary['total_events']}")
        print(f"Watch events: {summary['watch']}")
        print(f"Rate events: {summary['rate']}")
        print(f"Recommendation events: {summary['recommendation']}")
        print("Files created:")
        for path in summary["files_created"]:
            print(f"  - {path}")
    else:
        print("No new events were available.")

    if summary["parse_errors"]:
        print(f"Skipped {summary['parse_errors']} malformed events.", file=sys.stderr)


if __name__ == "__main__":
    main()
