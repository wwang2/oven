import pytest
from pathlib import Path
from oven_core.backends.local import LocalBackend
from oven_core.runtime import RunResult

def test_local_backend_submit(tmp_path):
    # Use a temporary directory for runs
    backend = LocalBackend(base_dir=tmp_path)
    
    # We need to make sure workloads are discoverable. 
    # Since we are in the root of the repo during tests, workloads.demo_task should be importable.
    run_id = backend.submit("demo_task", {"n": 5})
    
    assert run_id is not None
    status = backend.get_status(run_id)
    assert status.status == "succeeded"
    assert "result" in status.metadata
    assert status.metadata["result"] == 30  # sum(i*i for i in range(5)) = 0+1+4+9+16 = 30

def test_local_backend_invalid_task(tmp_path):
    backend = LocalBackend(base_dir=tmp_path)
    with pytest.raises(ValueError, match="not found"):
        backend.submit("non_existent_task", {})

