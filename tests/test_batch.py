"""Tests for batch/map execution."""

import pytest
from pathlib import Path

from oven_core.backends.local import LocalBackend


def test_submit_batch(tmp_path):
    """Test submitting multiple tasks in batch."""
    backend = LocalBackend(base_dir=tmp_path, max_workers=2)
    
    inputs_list = [
        {"n": 5},
        {"n": 10},
        {"n": 15},
    ]
    
    run_ids = backend.submit_batch("demo_task", inputs_list)
    
    assert len(run_ids) == 3
    assert all(isinstance(rid, str) for rid in run_ids)
    
    # All should have succeeded
    for run_id in run_ids:
        result = backend.get_status(run_id)
        assert result.status == "succeeded"


def test_map(tmp_path):
    """Test the map generator."""
    backend = LocalBackend(base_dir=tmp_path, max_workers=2)
    
    inputs_list = [
        {"n": 3},
        {"n": 6},
    ]
    
    results = list(backend.map("demo_task", inputs_list))
    
    assert len(results) == 2
    
    # Results are (run_id, RunResult) tuples
    all_succeeded = all(result.status == "succeeded" for _, result in results)
    assert all_succeeded
    
    # Check result values (they may be in any order due to parallelism)
    result_values = {result.metadata["result"] for _, result in results}
    expected = {sum(i*i for i in range(3)), sum(i*i for i in range(6))}  # 5, 55
    assert result_values == expected


def test_gather(tmp_path):
    """Test the gather method."""
    backend = LocalBackend(base_dir=tmp_path, max_workers=2)
    
    inputs_list = [
        {"n": 2},
        {"n": 4},
    ]
    
    batch_result = backend.gather("demo_task", inputs_list)
    
    assert len(batch_result.run_ids) == 2
    assert len(batch_result.results) == 2
    assert len(batch_result.succeeded()) == 2
    assert len(batch_result.failed()) == 0

