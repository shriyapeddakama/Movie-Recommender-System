"""Parsing rules - Single Source of Truth for event parsing logic"""

from typing import Dict
from pydantic import BaseModel


class ParsingRule(BaseModel):
    """Single parsing rule definition"""
    event_type: str
    pattern: str  # Regular expression pattern
    field_mappings: Dict[str, int]  # Field name -> regex group index


# Single Source of Truth for parsing rules
# These patterns are used by both Python and Spark parsers

PARSING_RULES = [
    ParsingRule(
        event_type="watch",
        pattern=r'GET /data/m/([^/]+)/(\d+)\.mpg',
        field_mappings={
            "movie_id": 1,
            "minute": 2
        }
    ),
    ParsingRule(
        event_type="rate",
        pattern=r'GET /rate/([^=]+)=(\d+)',
        field_mappings={
            "movie_id": 1,
            "rating": 2
        }
    ),
    ParsingRule(
        event_type="recommendation",
        pattern=r'recommendation request ([^,]+).*?status (\d+).*?result: ([^,]+)',
        field_mappings={
            "server": 1,
            "status_code": 2,
            "recommendations": 3
        }
    )
]
