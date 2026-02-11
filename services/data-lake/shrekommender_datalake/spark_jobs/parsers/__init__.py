"""Spark-based parsers for event data"""

from .spark_parser import parse_kafka_stream

__all__ = ["parse_kafka_stream"]
