import pytest
from shrekommender_system.data.schema import parse_event, WatchEvent, RateEvent, RecommendationEvent

@pytest.mark.parametrize(
    "line,expected_type,expected_fields",
    [
        (
            "2025-09-26T22:17:51,60121,GET /data/m/trances+1981/31.mpg",
            WatchEvent,
            {"timestamp": "2025-09-26T22:17:51", "user_id": "60121", "movie_id": "trances+1981", "minute": 31}
        ),
        (
            "2025-09-26T22:20:00,60121,GET /rate/trances+1981=4",
            RateEvent,
            {"timestamp": "2025-09-26T22:20:00", "user_id": "60121", "movie_id": "trances+1981", "rating": 4}
        ),
        (
            "2025-09-26T22:25:00,60121,recommendation request server1, status 200, result: [movie1,movie2], 123ms",
            RecommendationEvent,
            {"timestamp": "2025-09-26T22:25:00", "user_id": "60121", "server": "server1",
             "status_code": 200, "recommendations": "[movie1,movie2]", "response_time": " 123ms"}
        ),
        (
            "invalid,line,here",
            type(None),
            None
        )
    ]
)
def test_parse_event_parametrized(line, expected_type, expected_fields):
    event = parse_event(line)
    assert isinstance(event, expected_type)
    if expected_fields:
        for key, value in expected_fields.items():
            assert getattr(event, key) == value
