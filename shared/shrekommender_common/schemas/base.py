"""Base event schemas and types"""

from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Event type enumeration"""
    WATCH = "watch"
    RATE = "rate"
    RECOMMENDATION = "recommendation"


class BaseEvent(BaseModel):
    """Base class for all events"""
    timestamp: datetime = Field(description="Event timestamp")
    user_id: str = Field(description="User ID")
    event_type: EventType = Field(description="Event type")

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
