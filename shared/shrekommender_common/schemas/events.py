"""Event schemas - Single Source of Truth for all event definitions"""

from typing import Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field
from .base import BaseEvent, EventType


class WatchEvent(BaseEvent):
    """Movie watching event

    Format: <time>,<userid>,GET /data/m/<movieid>/<minute>.mpg
    """
    event_type: Literal[EventType.WATCH] = EventType.WATCH
    movie_id: str = Field(description="Movie ID")
    minute: int = Field(ge=0, description="Minute of the movie being watched")


class RateEvent(BaseEvent):
    """Movie rating event

    Format: <time>,<userid>,GET /rate/<movieid>=<rating>
    """
    event_type: Literal[EventType.RATE] = EventType.RATE
    movie_id: str = Field(description="Movie ID")
    rating: int = Field(ge=1, le=5, description="Rating from 1 to 5")


class RecommendationEvent(BaseEvent):
    """Recommendation request event

    Format: <time>,<userid>,recommendation request <server>, status <code>, result: <recommendations>, <responsetime>
    """
    event_type: Literal[EventType.RECOMMENDATION] = EventType.RECOMMENDATION
    server: str = Field(description="Recommendation server name")
    status_code: int = Field(description="HTTP status code")
    recommendations: str = Field(description="Comma-separated list of recommended movie IDs")
    response_time: Optional[str] = Field(None, description="Response time")


# ========== Unified Wide Table Schema (Single Source of Truth) ==========

class UnifiedEventRecord(BaseModel):
    """
    Unified event wide table schema

    This is the SINGLE SOURCE OF TRUTH for the data lake schema.
    All format conversions (Spark StructType, etc.) are auto-generated from this.
    """
    # ===== Common fields =====
    event_id: str = Field(description="Unique event ID (UUID)")
    timestamp: datetime = Field(description="Event timestamp")
    date: str = Field(description="Event date for partitioning (yyyy-MM-dd)")
    user_id: str = Field(description="User ID")
    event_type: EventType = Field(description="Event type: watch | rate | recommendation")

    # ===== Watch event fields =====
    movie_id: Optional[str] = Field(None, description="Movie ID (for watch and rate events)")
    minute: Optional[int] = Field(None, ge=0, description="Minute watched (for watch events)")

    # ===== Rate event fields =====
    rating: Optional[int] = Field(None, ge=1, le=5, description="Rating 1-5 (for rate events)")

    # ===== Recommendation event fields =====
    server: Optional[str] = Field(None, description="Recommendation server (for recommendation events)")
    status_code: Optional[int] = Field(None, description="HTTP status code (for recommendation events)")
    recommendations: Optional[str] = Field(None, description="Recommended movie IDs (for recommendation events)")
    response_time: Optional[str] = Field(None, description="Response time (for recommendation events)")

    # ===== Metadata =====
    ingestion_time: datetime = Field(description="Data ingestion timestamp")
    source_topic: str = Field(description="Source Kafka topic")

    class Config:
        use_enum_values = True

    @classmethod
    def from_watch_event(cls, event: WatchEvent, **metadata) -> "UnifiedEventRecord":
        """Convert from WatchEvent to unified format"""
        import uuid
        return cls(
            event_id=str(uuid.uuid4()),
            timestamp=event.timestamp,
            date=event.timestamp.strftime("%Y-%m-%d"),
            user_id=event.user_id,
            event_type=EventType.WATCH,
            movie_id=event.movie_id,
            minute=event.minute,
            **metadata
        )

    @classmethod
    def from_rate_event(cls, event: RateEvent, **metadata) -> "UnifiedEventRecord":
        """Convert from RateEvent to unified format"""
        import uuid
        return cls(
            event_id=str(uuid.uuid4()),
            timestamp=event.timestamp,
            date=event.timestamp.strftime("%Y-%m-%d"),
            user_id=event.user_id,
            event_type=EventType.RATE,
            movie_id=event.movie_id,
            rating=event.rating,
            **metadata
        )

    @classmethod
    def from_recommendation_event(cls, event: RecommendationEvent, **metadata) -> "UnifiedEventRecord":
        """Convert from RecommendationEvent to unified format"""
        import uuid
        return cls(
            event_id=str(uuid.uuid4()),
            timestamp=event.timestamp,
            date=event.timestamp.strftime("%Y-%m-%d"),
            user_id=event.user_id,
            event_type=EventType.RECOMMENDATION,
            server=event.server,
            status_code=event.status_code,
            recommendations=event.recommendations,
            response_time=event.response_time,
            **metadata
        )
