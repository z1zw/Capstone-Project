from app.clean.clean import CleanConnection
import pytest


def test_clean_connection_valid():
    obj = CleanConnection(
        entities={"source": 1, "target": 2},
        groups={"neuropil": "AL"},
        counts={"syn": 10},
        metrics={"weight": 0.5},
        stats={"metric_entropy": 0.2},
    )
    assert obj.entities["source"] == 1


def test_clean_connection_invalid_type():
    with pytest.raises(Exception):
        CleanConnection(
            entities={"source": "bad"},  
            groups={},
            counts={},
            metrics={},
            stats={},
        )
