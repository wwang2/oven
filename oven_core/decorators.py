"""
Decorator-based API for defining Oven tasks.

This provides a simpler, more Pythonic way to define tasks compared to the
class-based BaseTask approach.

Example:
    from oven_core.decorators import task, ImageSpec
    
    @task(cpu=0.5, memory=512, gpu="A10G")
    def train_model(context, learning_rate: float = 0.01):
        # Your training logic here
        return {"loss": 0.1}
    
    # With custom image
    @task(
        cpu=1.0,
        image=ImageSpec(
            pip_packages=["torch", "transformers"],
            apt_packages=["git"],
        )
    )
    def inference_task(context, prompt: str):
        ...
"""

from typing import Any, Callable, Dict, List, Optional, Union
from functools import wraps
from pydantic import BaseModel, Field

from oven_core.runtime import BaseTask, WorkloadSpec, RunContext


class ImageSpec(BaseModel):
    """Specification for a custom container image."""
    
    base: str = "debian_slim"
    python_version: str = "3.11"
    pip_packages: List[str] = Field(default_factory=list)
    apt_packages: List[str] = Field(default_factory=list)
    env_vars: Dict[str, str] = Field(default_factory=dict)
    
    # For advanced users: pre-built image reference
    from_registry: Optional[str] = None
    
    def to_modal_image(self):
        """Convert to a Modal Image object."""
        import modal
        
        if self.from_registry:
            img = modal.Image.from_registry(
                self.from_registry, 
                add_python=self.python_version
            )
        elif self.base == "debian_slim":
            img = modal.Image.debian_slim(python_version=self.python_version)
        elif self.base == "micromamba":
            img = modal.Image.micromamba(python_version=self.python_version)
        else:
            img = modal.Image.debian_slim(python_version=self.python_version)
        
        if self.apt_packages:
            img = img.apt_install(*self.apt_packages)
        
        if self.pip_packages:
            img = img.pip_install(*self.pip_packages)
        
        if self.env_vars:
            img = img.env(self.env_vars)
        
        return img


class FunctionTask(BaseTask):
    """Wrapper that adapts a decorated function to the BaseTask interface."""
    
    def __init__(self, fn: Callable, workload_spec: WorkloadSpec, image_spec: Optional[ImageSpec] = None):
        self._fn = fn
        self._spec = workload_spec
        self._image_spec = image_spec
    
    @property
    def spec(self) -> WorkloadSpec:
        return self._spec
    
    @property
    def image_spec(self) -> Optional[ImageSpec]:
        return self._image_spec
    
    def run(self, context: RunContext, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return self._fn(context, **inputs)


# Global registry of decorated tasks
_task_registry: Dict[str, FunctionTask] = {}


def get_task(name: str) -> Optional[FunctionTask]:
    """Get a registered task by name."""
    return _task_registry.get(name)


def list_tasks() -> List[str]:
    """List all registered task names."""
    return list(_task_registry.keys())


def task(
    name: Optional[str] = None,
    cpu: float = 0.1,
    memory: int = 128,
    gpu: Optional[str] = None,
    timeout: int = 600,
    image: Optional[ImageSpec] = None,
    retries: int = 0,
):
    """
    Decorator to define a workload task.
    
    Args:
        name: Task name (defaults to function name)
        cpu: CPU cores to allocate
        memory: Memory in MB
        gpu: GPU type (e.g., "T4", "A10G", "A100", "H100")
        timeout: Timeout in seconds
        image: Custom ImageSpec for container configuration
        retries: Number of retries on failure
    
    Example:
        @task(cpu=0.5, memory=512)
        def my_task(context, param1: int, param2: str = "default"):
            result = do_something(param1, param2)
            return {"output": result}
    """
    def decorator(fn: Callable) -> Callable:
        task_name = name or fn.__name__
        
        workload_spec = WorkloadSpec(
            name=task_name,
            cpu=cpu,
            memory=memory,
            gpu=gpu,
            timeout=timeout,
        )
        
        # Create the FunctionTask wrapper
        fn_task = FunctionTask(fn, workload_spec, image)
        
        # Register in the global registry
        _task_registry[task_name] = fn_task
        
        # Attach metadata to the function for introspection
        fn._oven_task = fn_task
        fn._oven_spec = workload_spec
        fn._oven_image = image
        fn._oven_retries = retries
        
        @wraps(fn)
        def wrapper(*args, **kwargs):
            # Direct call just runs the function
            return fn(*args, **kwargs)
        
        # Copy the task metadata to wrapper
        wrapper._oven_task = fn_task
        wrapper._oven_spec = workload_spec
        wrapper._oven_image = image
        wrapper._oven_retries = retries
        
        return wrapper
    
    return decorator


# Convenience function to create ImageSpec
def image(
    pip_packages: Optional[List[str]] = None,
    apt_packages: Optional[List[str]] = None,
    python_version: str = "3.11",
    base: str = "debian_slim",
    env_vars: Optional[Dict[str, str]] = None,
    from_registry: Optional[str] = None,
) -> ImageSpec:
    """
    Create an ImageSpec for task configuration.
    
    Example:
        @task(
            image=image(
                pip_packages=["torch", "numpy"],
                apt_packages=["ffmpeg"],
            )
        )
        def my_gpu_task(context):
            ...
    """
    return ImageSpec(
        base=base,
        python_version=python_version,
        pip_packages=pip_packages or [],
        apt_packages=apt_packages or [],
        env_vars=env_vars or {},
        from_registry=from_registry,
    )

