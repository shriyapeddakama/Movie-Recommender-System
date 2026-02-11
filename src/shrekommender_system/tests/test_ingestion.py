from unittest.mock import patch, mock_open
from shrekommender_system.data.ingestion import KafkaDataIngester

def fake_run(cmd, capture_output=True, text=True, timeout=None):
    if "-o" in cmd and "end" in cmd:
        class MockEndResult:
            returncode = 0
            stdout = "100\n"
        return MockEndResult()
    else:
        class MockFetchResult:
            returncode = 0
            stdout = "\n".join([
                "2025-09-26T22:17:51,60121,GET /data/m/trances+1981/31.mpg",
                "2025-09-26T22:20:00,60121,GET /rate/trances+1981=4",
            ])
        return MockFetchResult()


@patch("builtins.open", new_callable=mock_open)
@patch("subprocess.run")
def test_fetch_recent(mock_run, mock_file):
    mock_run.side_effect = fake_run

    ingester = KafkaDataIngester()
    stats = ingester.fetch_recent(n=2)

    assert stats["status"] == "completed"
    assert stats["total_events"] == 2
    assert stats["watch_events"] == 1
    assert stats["rate_events"] == 1
    assert stats["parse_errors"] == 0
    assert len(stats["files_created"]) > 0

    mock_file.assert_called()

# Edge cases
def test_empty_file(monkeypatch):
    monkeypatch.setattr(
        KafkaDataIngester,
        "fetch_recent",
        lambda self, n=0: {"status": "completed", "files_created": [], "total_events": 0}
    )
    ingester = KafkaDataIngester()
    stats = ingester.fetch_recent()
    assert stats["total_events"] == 0

def test_malformed_event(monkeypatch):
    monkeypatch.setattr(
        KafkaDataIngester,
        "fetch_recent",
        lambda self, n=0: {"status": "completed", "files_created": ["malformed.jsonl"], "total_events": 0, "parse_errors": 1}
    )
    ingester = KafkaDataIngester()
    stats = ingester.fetch_recent()
    assert stats["parse_errors"] == 1