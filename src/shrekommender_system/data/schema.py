"""Data schemas for the shrekommender system"""

from dataclasses import dataclass
from typing import Optional
import re


@dataclass
class WatchEvent:
    """Movie watching event
    Format: <time>,<userid>,GET /data/m/<movieid>/<minute>.mpg
    """
    timestamp: str
    user_id: str
    movie_id: str
    minute: int

    @property
    def date(self) -> str:
        return self.timestamp.split('T')[0]

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "event_type": "watch",
            "movie_id": self.movie_id,
            "minute": self.minute
        }


@dataclass
class RateEvent:
    """Movie rating event
    Format: <time>,<userid>,GET /rate/<movieid>=<rating>
    """
    timestamp: str
    user_id: str
    movie_id: str
    rating: int

    @property
    def date(self) -> str:
        return self.timestamp.split('T')[0]

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "event_type": "rate",
            "movie_id": self.movie_id,
            "rating": self.rating
        }


@dataclass
class RecommendationEvent:
    """Recommendation request event
    Format: <time>,<userid>,recommendation request <server>, status <code>, result: <recommendations>, <responsetime>
    """
    timestamp: str
    user_id: str
    server: str
    status_code: int
    recommendations: str
    response_time: Optional[str] = None

    @property
    def date(self) -> str:
        return self.timestamp.split('T')[0]

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "event_type": "recommendation",
            "server": self.server,
            "status_code": self.status_code,
            "recommendations": self.recommendations,
            "response_time": self.response_time
        }


def parse_event(line: str):
    """Parse a CSV line from Kafka into the appropriate event type"""
    try:
        parts = line.strip().split(',', 2)
        if len(parts) < 3:
            return None

        timestamp = parts[0]
        user_id = parts[1]
        action = parts[2]

        # Watch event
        if action.startswith("GET /data/m/"):
            match = re.match(r'GET /data/m/([^/]+)/(\d+)\.mpg', action)
            if match:
                return WatchEvent(
                    timestamp=timestamp,
                    user_id=user_id,
                    movie_id=match.group(1),
                    minute=int(match.group(2))
                )

        # Rate event
        elif action.startswith("GET /rate/"):
            match = re.match(r'GET /rate/([^=]+)=(\d+)', action)
            if match:
                return RateEvent(
                    timestamp=timestamp,
                    user_id=user_id,
                    movie_id=match.group(1),
                    rating=int(match.group(2))
                )

        # Recommendation event
        elif "recommendation request" in action:
            # Parse the rest of the line for recommendation details
            full_line = line.strip()

            server_match = re.search(r'recommendation request ([^,]+)', full_line)
            status_match = re.search(r'status (\d+)', full_line)
            result_match = re.search(r'result: (\[.*\])', full_line)


            # Response time is the last field
            parts = full_line.split(',')
            response_time = parts[-1] if len(parts) > 5 else None

            if server_match and status_match:
                return RecommendationEvent(
                    timestamp=timestamp,
                    user_id=user_id,
                    server=server_match.group(1).strip(),
                    status_code=int(status_match.group(1)),
                    recommendations=result_match.group(1).strip() if result_match else "",
                    response_time=response_time
                )

        return None

    except Exception:
        return None