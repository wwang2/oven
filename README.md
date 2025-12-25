# Oven

Serverless cloud jobs, and it does not have to be intimidating

## Three-layer separation

1.  **Workload layer**: Pure business logic, no infra imports.
2.  **Runtime layer**: Backend-agnostic execution semantics.
3.  **Backend layer**: Adaptations for Modal, RunPod, or Local execution.

## Installation

```bash
# Clone the repo
git clone <repo-url>
cd oven

# Install as an editable package (makes 'workloads' and 'oven_core' importable)
pip install -e .
```

## Quick Start

### Decorator-based tasks (recommended)

```python
# workloads/my_task.py
from oven_core import task, ImageSpec, RunContext

@task(cpu=0.5, memory=256)
def simple_task(context: RunContext, message: str = "Hello") -> dict:
    print(f"Processing: {message}")
    return {"message": message, "status": "done"}

# With custom dependencies
@task(
    cpu=1.0,
    memory=512,
    gpu="A10G",
    image=ImageSpec(
        pip_packages=["torch", "transformers"],
        python_version="3.11",
    ),
)
def ml_task(context: RunContext, prompt: str) -> dict:
    import torch
    # Your ML code here
    return {"device": "cuda" if torch.cuda.is_available() else "cpu"}
```

### Class-based tasks (original pattern)

```python
# workloads/demo_task.py
from oven_core import BaseTask, WorkloadSpec, RunContext

class DemoTask(BaseTask):
    @property
    def spec(self) -> WorkloadSpec:
        return WorkloadSpec(name="demo_task", cpu=0.1, memory=128)

    def run(self, context: RunContext, inputs: dict) -> dict:
        n = inputs.get("n", 100)
        result = sum(i * i for i in range(n))
        return {"n": n, "result": result}
```

## Usage

### Run a single task

```bash
# Run locally
python cli.py run demo_task --backend local -i n=100

# Run on Modal
python cli.py run demo_task --backend modal -i n=100
```

### Map over multiple inputs (hyperparameter sweep)

```bash
# Sweep over a single parameter
python cli.py map demo_task -p n -v "[10, 50, 100, 500]" --backend local

# Sweep with fixed parameters
python cli.py map sweep_task -p learning_rate -v "[0.001, 0.01, 0.1]" -i epochs=50

# Run on Modal with parallelism
python cli.py map sweep_task -p batch_size -v "[16, 32, 64, 128]" --backend modal
```

### Check status and fetch artifacts

```bash
# Check status
python cli.py status <run_id> --backend local

# Fetch artifacts
python cli.py fetch <run_id> --backend modal -d ./downloads
```

## Custom Images

Define per-task dependencies with `ImageSpec`:

```python
from oven_core import task, ImageSpec

@task(
    cpu=1.0,
    gpu="H100",
    image=ImageSpec(
        # Base image options
        base="debian_slim",              # or "micromamba"
        python_version="3.11",
        
        # Dependencies
        pip_packages=["torch", "numpy", "transformers"],
        apt_packages=["ffmpeg", "git"],
        
        # Environment variables
        env_vars={"HF_HOME": "/cache/huggingface"},
        
        # Or use a pre-built image
        # from_registry="nvidia/cuda:12.0-devel-ubuntu22.04",
    ),
)
def gpu_task(context, prompt: str):
    ...
```

## CLI Reference

```bash
# Run a single task
python cli.py run <task_name> --backend <local|modal> -i key=value

# Map over multiple inputs
python cli.py map <task_name> -p <param> -v '[values]' --backend <local|modal>

# Deploy to Modal
python cli.py deploy

# Check run status
python cli.py status <run_id> --backend <local|modal>

# Fetch artifacts
python cli.py fetch <run_id> --backend <local|modal> -d <dest_dir>
```

## GitHub Actions

This template includes a **Deploy** workflow that automatically redeploys the Modal app on every push to `main` or `master`.

### Setup Secrets

For the deployment to work, you must add the following Modal secrets to your GitHub repository:
- `MODAL_TOKEN_ID`: Your Modal Token ID.
- `MODAL_TOKEN_SECRET`: Your Modal Token Secret.

You can generate these in your [Modal Settings](https://modal.com/settings/tokens).

## Roadmap

- [x] **Function Decorator API**: Simpler `@task` decorator pattern
- [x] **Parallel Map/Batch**: Run multiple tasks in parallel
- [x] **Per-task Images**: Custom dependencies per task
- [ ] **RunPod Backend**: Support for serverless GPU workloads on RunPod.
- [ ] **AWS Backend**: Support for AWS Batch or Lambda.
- [ ] **GCP Backend**: Support for Google Cloud Run or Vertex AI.
- [ ] **Kubernetes Backend**: Support for standard K8s jobs.
- [ ] **Agent Tooling**: Specialized JSON schemas and simplified polling for AI agents.
- [ ] **Durable Metadata Store**: Move metadata from local JSON files to a shared DB (e.g., PostgreSQL or Supabase).
