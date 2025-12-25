"""Tests for decorator-based task definitions."""

import pytest
from pathlib import Path

from oven_core.decorators import task, ImageSpec, get_task, list_tasks, _task_registry
from oven_core.runtime import RunContext


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear the task registry before each test."""
    _task_registry.clear()
    yield
    _task_registry.clear()


def test_task_decorator_basic():
    """Test that the @task decorator registers a task."""
    
    @task(cpu=0.5, memory=256)
    def my_test_task(context: RunContext, value: int = 10):
        return {"value": value, "doubled": value * 2}
    
    # Check it's registered
    assert "my_test_task" in list_tasks()
    
    # Get the task
    fn_task = get_task("my_test_task")
    assert fn_task is not None
    assert fn_task.spec.cpu == 0.5
    assert fn_task.spec.memory == 256


def test_task_decorator_with_custom_name():
    """Test that the @task decorator supports custom names."""
    
    @task(name="custom_name", cpu=0.1)
    def internal_function(context: RunContext):
        return {"status": "done"}
    
    assert "custom_name" in list_tasks()
    assert "internal_function" not in list_tasks()


def test_task_decorator_with_image():
    """Test that the @task decorator supports ImageSpec."""
    
    img = ImageSpec(
        pip_packages=["numpy", "pandas"],
        apt_packages=["git"],
        python_version="3.10",
    )
    
    @task(cpu=1.0, image=img)
    def ml_task(context: RunContext):
        return {}
    
    fn_task = get_task("ml_task")
    assert fn_task.image_spec is not None
    assert "numpy" in fn_task.image_spec.pip_packages
    assert "pandas" in fn_task.image_spec.pip_packages
    assert fn_task.image_spec.python_version == "3.10"


def test_task_execution(tmp_path):
    """Test that a decorated task can be executed."""
    
    @task(cpu=0.1)
    def executable_task(context: RunContext, x: int, y: int = 5):
        return {"sum": x + y, "product": x * y}
    
    fn_task = get_task("executable_task")
    
    context = RunContext(
        run_id="test-123",
        work_dir=tmp_path,
        artifact_path=tmp_path / "artifacts",
    )
    (tmp_path / "artifacts").mkdir()
    
    result = fn_task.run(context, {"x": 3, "y": 7})
    
    assert result["sum"] == 10
    assert result["product"] == 21


def test_image_spec_defaults():
    """Test ImageSpec default values."""
    img = ImageSpec()
    
    assert img.base == "debian_slim"
    assert img.python_version == "3.11"
    assert img.pip_packages == []
    assert img.apt_packages == []
    assert img.env_vars == {}
    assert img.from_registry is None


def test_image_spec_from_registry():
    """Test ImageSpec with a custom registry image."""
    img = ImageSpec(
        from_registry="nvidia/cuda:12.0-devel-ubuntu22.04",
        python_version="3.11",
        pip_packages=["torch"],
    )
    
    assert img.from_registry == "nvidia/cuda:12.0-devel-ubuntu22.04"
    assert "torch" in img.pip_packages

