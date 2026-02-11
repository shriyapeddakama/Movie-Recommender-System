"""Data schemas for the shrekommender system"""

from .base import EventType, BaseEvent
from .events import WatchEvent, RateEvent, RecommendationEvent, UnifiedEventRecord
from .parsing_rules import ParsingRule, PARSING_RULES

__all__ = [
    "EventType",
    "BaseEvent",
    "WatchEvent",
    "RateEvent",
    "RecommendationEvent",
    "UnifiedEventRecord",
    "ParsingRule",
    "PARSING_RULES",
]

# Converters are imported separately to avoid circular dependencies
# and because they require PySpark which may not always be available
# Use: from shrekommender_common.schemas.converters import pydantic_to_spark_schema
