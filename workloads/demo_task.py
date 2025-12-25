"""
Demo workloads showcasing both class-based and decorator-based task definitions.

This module demonstrates:
1. Class-based tasks (original pattern)
2. Decorator-based tasks (new simpler pattern)
3. Custom image configuration per task
"""

from typing import Any, Dict
import time

from oven_core.runtime import BaseTask, WorkloadSpec, RunContext
from oven_core.decorators import task, ImageSpec


# =============================================================================
# CLASS-BASED TASK (Original Pattern)
# =============================================================================

class DemoTask(BaseTask):
    """
    A simple demo task using the class-based pattern.
    
    This is the original pattern that provides full control over
    the task specification and execution.
    """
    
    @property
    def spec(self) -> WorkloadSpec:
        return WorkloadSpec(
            name="demo_task",
            cpu=0.1,
            memory=128,
        )

    def run(self, context: RunContext, inputs: Dict[str, Any]) -> Dict[str, Any]:
        n = inputs.get("n", 100)
        print(f"Running demo task with n={n}...")
        
        # Simulate some work
        time.sleep(1)
        
        result = sum(i * i for i in range(n))
        
        # Write an artifact
        artifact_file = context.artifact_path / "result.txt"
        with open(artifact_file, "w") as f:
            f.write(f"Sum of squares up to {n} is {result}\n")
            
        return {
            "n": n,
            "result": result,
            "artifact_file": str(artifact_file.name),
        }


# =============================================================================
# DECORATOR-BASED TASKS (New Simpler Pattern)
# =============================================================================

@task(cpu=0.1, memory=128)
def simple_task(context: RunContext, message: str = "Hello") -> Dict[str, Any]:
    """
    A minimal task using the decorator pattern.
    
    Usage:
        python cli.py run simple_task -i message="Hello World"
    """
    print(f"Simple task received: {message}")
    
    # Write output
    output_file = context.artifact_path / "output.txt"
    with open(output_file, "w") as f:
        f.write(f"Message: {message}\n")
    
    return {"message": message, "status": "completed"}


@task(
    cpu=0.5,
    memory=256,
    image=ImageSpec(
        pip_packages=["numpy", "scipy"],
        python_version="3.11",
    ),
)
def compute_task(context: RunContext, size: int = 1000) -> Dict[str, Any]:
    """
    A compute task with custom dependencies (numpy, scipy).
    
    This demonstrates how to specify per-task dependencies that will
    be installed in the container image.
    
    Usage:
        python cli.py run compute_task -i size=5000
        python cli.py map compute_task -p size -v "[100, 500, 1000, 5000]"
    """
    import numpy as np
    from scipy import stats
    
    print(f"Running compute task with size={size}...")
    
    # Generate random data and compute statistics
    data = np.random.randn(size)
    
    result = {
        "size": size,
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "skewness": float(stats.skew(data)),
        "kurtosis": float(stats.kurtosis(data)),
    }
    
    # Save to artifact
    artifact_file = context.artifact_path / "stats.json"
    import json
    with open(artifact_file, "w") as f:
        json.dump(result, f, indent=2)
    
    return result


@task(
    cpu=1.0,
    memory=512,
    gpu="T4",  # Request a T4 GPU
    timeout=1800,
    image=ImageSpec(
        pip_packages=["torch", "transformers"],
        python_version="3.11",
    ),
)
def gpu_task(context: RunContext, prompt: str = "Hello, world!") -> Dict[str, Any]:
    """
    A GPU task example with PyTorch and Transformers.
    
    This shows how to:
    - Request GPU resources
    - Use custom pip packages (torch, transformers)
    - Set a longer timeout for ML workloads
    
    Usage:
        python cli.py run gpu_task --backend modal -i prompt="Explain quantum computing"
    """
    import torch
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")
    print(f"Prompt: {prompt}")
    
    # Simple tensor operation to verify GPU works
    x = torch.randn(1000, 1000, device=device)
    y = torch.matmul(x, x.T)
    
    result = {
        "device": device,
        "prompt": prompt,
        "tensor_shape": list(y.shape),
        "cuda_available": torch.cuda.is_available(),
    }
    
    # For a real ML task, you would load a model and run inference here
    # from transformers import pipeline
    # generator = pipeline("text-generation", model="gpt2", device=device)
    # output = generator(prompt, max_length=100)
    
    return result


# =============================================================================
# HYPERPARAMETER SWEEP EXAMPLE
# =============================================================================

@task(
    cpu=0.25,
    memory=256,
    image=ImageSpec(pip_packages=["numpy"]),
)
def sweep_task(
    context: RunContext,
    learning_rate: float = 0.01,
    batch_size: int = 32,
    epochs: int = 10,
) -> Dict[str, Any]:
    """
    Example task for hyperparameter sweeps.
    
    This can be used with the map command to sweep over parameters:
    
    Usage:
        python cli.py map sweep_task -p learning_rate -v "[0.001, 0.01, 0.1]" -i epochs=50
        python cli.py map sweep_task -p batch_size -v "[16, 32, 64, 128]"
    """
    import numpy as np
    
    print(f"Training with lr={learning_rate}, batch_size={batch_size}, epochs={epochs}")
    
    # Simulate training - in reality this would train a model
    np.random.seed(int(learning_rate * 10000) + batch_size)
    
    # Simulate that some hyperparameters are better than others
    base_loss = 1.0
    lr_effect = -0.5 * np.log10(learning_rate + 1e-4)  # Lower LR generally better
    batch_effect = 0.1 * np.log2(batch_size / 32)  # Larger batches slightly worse
    noise = np.random.randn() * 0.1
    
    final_loss = base_loss + lr_effect + batch_effect + noise
    final_loss = max(0.01, final_loss)  # Clamp to reasonable values
    
    time.sleep(0.5)  # Simulate training time
    
    result = {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "final_loss": round(float(final_loss), 4),
        "converged": bool(final_loss < 0.5),
    }
    
    # Save training curve artifact
    artifact_file = context.artifact_path / "training_result.json"
    import json
    with open(artifact_file, "w") as f:
        json.dump(result, f, indent=2)
    
    return result
