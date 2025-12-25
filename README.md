# Oven

Serverless cloud jobs, made simple. modify \workloads to \you-project-name if you want.

## Installation

```bash
git clone <repo-url>
cd oven
pip install -e .
```

## Setup Modal

1. **Install Modal CLI**:
   ```bash
   pip install modal
   ```

2. **Authenticate**:
   ```bash
   modal token new
   ```
   This will open your browser to create a token. Copy the token ID and secret.

3. **Set environment variables** (optional, for CI/CD):
   ```bash
   export MODAL_TOKEN_ID="your-token-id"
   export MODAL_TOKEN_SECRET="your-token-secret"
   ```

## Quick Start

### Define a task

```python
# workloads/my_task.py
from oven_core import task, RunContext

@task(cpu=0.5, memory=256)
def my_task(context: RunContext, n: int = 100) -> dict:
    result = sum(i * i for i in range(n))
    return {"result": result}
```

### Run it

```bash
# Local execution
python cli.py run my_task --backend local -i n=50

# On Modal
python cli.py run my_task --backend modal -i n=50
```

### Parallel execution

```bash
# Map over multiple values
python cli.py map my_task -p n -v "[10, 50, 100, 500]" --backend modal
```

## Custom Dependencies

```python
from oven_core import task, ImageSpec

@task(
    gpu="A10G",
    image=ImageSpec(
        pip_packages=["torch", "transformers"],
        python_version="3.11",
    ),
)
def ml_task(context: RunContext, prompt: str) -> dict:
    import torch
    return {"device": "cuda" if torch.cuda.is_available() else "cpu"}
```

## CLI Commands

```bash
python cli.py run <task> --backend <local|modal> -i key=value
python cli.py map <task> -p <param> -v '[values]' --backend <local|modal>
python cli.py status <run_id> --backend <local|modal>
python cli.py fetch <run_id> --backend <local|modal> -d <dest>
python cli.py deploy  # Deploy Modal app
```

## Architecture

- **Workload layer**: Pure business logic (no infra imports)
- **Runtime layer**: Backend-agnostic execution semantics
- **Backend layer**: Modal, Local (RunPod, AWS, GCP coming soon)
