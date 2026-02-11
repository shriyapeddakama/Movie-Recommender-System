"""Spark SQL parser for Kafka events

This parser uses Spark SQL native functions for optimal performance.
It reads parsing rules from shrekommender_common (Single Source of Truth).
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    regexp_extract, split, col, when,
    to_timestamp, to_date, lit, current_timestamp,
    expr, trim, element_at
)
from shrekommender_common.schemas.parsing_rules import PARSING_RULES


def parse_kafka_stream(kafka_df: DataFrame) -> DataFrame:
    """
    Parse Kafka stream into unified event table format

    Uses parsing rules from shrekommender_common (Single Source of Truth)
    and converts to the unified wide table schema.

    Args:
        kafka_df: Kafka stream DataFrame with 'value' column

    Returns:
        DataFrame with unified event schema
    """
    # Get parsing rules
    watch_rule = next(r for r in PARSING_RULES if r.event_type == "watch")
    rate_rule = next(r for r in PARSING_RULES if r.event_type == "rate")
    rec_rule = next(r for r in PARSING_RULES if r.event_type == "recommendation")

    # Convert Kafka value to string and split CSV
    df = kafka_df.select(
        col("value").cast("string").alias("raw_line"),
        col("timestamp").alias("kafka_timestamp")
    )

    # Split CSV line: timestamp, user_id, action
    # Using limit=3 to handle commas in the action part
    df = df.withColumn("parts", split(col("raw_line"), ",", 3))

    # Extract common fields
    df = df.select(
        col("raw_line"),
        col("kafka_timestamp"),
        element_at(col("parts"), 1).alias("timestamp_str"),
        element_at(col("parts"), 2).alias("user_id"),
        element_at(col("parts"), 3).alias("action")
    )

    # Determine event type
    df = df.withColumn(
        "event_type",
        when(col("action").contains("GET /data/m/"), lit("watch"))
        .when(col("action").contains("GET /rate/"), lit("rate"))
        .when(col("action").contains("recommendation request"), lit("recommendation"))
        .otherwise(lit(None))
    )

    # Filter out unparseable events
    df = df.filter(col("event_type").isNotNull())

    # Parse event-specific fields
    parsed_df = df.select(
        # Generate unique event ID
        expr("uuid()").alias("event_id"),

        # Common fields (use try_cast to tolerate malformed timestamps)
        expr("try_to_timestamp(timestamp_str)").alias("timestamp"),
        expr("try_to_date(timestamp_str)").alias("date"),
        trim(col("user_id")).alias("user_id"),
        col("event_type"),

        # Watch event fields
        when(
            col("event_type") == "watch",
            trim(regexp_extract(col("action"), watch_rule.pattern, watch_rule.field_mappings["movie_id"]))
        ).otherwise(lit(None)).alias("movie_id_watch"),

        when(
            col("event_type") == "watch",
            expr("try_cast(regexp_extract(action, '{}', {}) as int)".format(
                watch_rule.pattern.replace("\\", "\\\\"),
                watch_rule.field_mappings['minute']
            ))
        ).otherwise(lit(None)).alias("minute"),

        # Rate event fields
        when(
            col("event_type") == "rate",
            trim(regexp_extract(col("action"), rate_rule.pattern, rate_rule.field_mappings["movie_id"]))
        ).otherwise(lit(None)).alias("movie_id_rate"),

        when(
            col("event_type") == "rate",
            expr("try_cast(regexp_extract(action, '{}', {}) as int)".format(
                rate_rule.pattern.replace("\\", "\\\\"),
                rate_rule.field_mappings['rating']
            ))
        ).otherwise(lit(None)).alias("rating"),

        # Recommendation event fields
        when(
            col("event_type") == "recommendation",
            trim(regexp_extract(col("raw_line"), rec_rule.pattern, rec_rule.field_mappings["server"]))
        ).otherwise(lit(None)).alias("server"),

        when(
            col("event_type") == "recommendation",
            expr("try_cast(regexp_extract(raw_line, '{}', {}) as int)".format(
                rec_rule.pattern.replace("\\", "\\\\"),
                rec_rule.field_mappings['status_code']
            ))
        ).otherwise(lit(None)).alias("status_code"),

        when(
            col("event_type") == "recommendation",
            trim(regexp_extract(col("raw_line"), rec_rule.pattern, rec_rule.field_mappings["recommendations"]))
        ).otherwise(lit(None)).alias("recommendations"),

        # Response time is the last field after the last comma for recommendation events
        when(
            col("event_type") == "recommendation",
            trim(element_at(split(col("raw_line"), ","), -1))
        ).otherwise(lit(None)).alias("response_time"),

        # Metadata
        current_timestamp().alias("ingestion_time"),
        lit("movielog11").alias("source_topic")  # TODO: Make this configurable
    )

    # Merge movie_id columns (watch and rate both have movie_id)
    final_df = parsed_df.select(
        col("event_id"),
        col("timestamp"),
        col("date"),
        col("user_id"),
        col("event_type"),

        # Coalesce movie_id from watch and rate events
        when(col("event_type") == "watch", col("movie_id_watch"))
        .when(col("event_type") == "rate", col("movie_id_rate"))
        .otherwise(lit(None)).alias("movie_id"),

        col("minute"),
        col("rating"),
        col("server"),
        col("status_code"),
        col("recommendations"),
        col("response_time"),
        col("ingestion_time"),
        col("source_topic")
    )

    return final_df
