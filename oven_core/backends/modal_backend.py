import os
import uuid
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Iterator, Tuple
import modal

from oven_core.runtime import (
    BaseBackend, BaseTask, RunContext, RunResult, WorkloadSpec, BatchResult
)

# Modal Setup
app = modal.App("oven-workloads")
volume = modal.Volume.from_name("oven-artifacts", create_if_missing=True)


def get_base_image() -> modal.Image:
    """Get the base image with oven_core and workloads."""
    return (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install("pydantic", "rich")
        .add_local_python_source("oven_core")
        .add_local_python_source("workloads")
    )


def get_task_image(task_name: str) -> modal.Image:
    """
    Build an image for a specific task, including any custom dependencies.
    
    This checks if the task has a custom ImageSpec and builds accordingly.
    """
    base = get_base_image()
    
    # Try to get task-specific image configuration
    try:
        import importlib
        module = importlib.import_module(f"workloads.{task_name}")
        
        # Check for decorator-based task with custom image
        from oven_core.decorators import get_task
        fn_task = get_task(task_name)
        
        if fn_task and fn_task.image_spec:
            image_spec = fn_task.image_spec
            
            # Start from custom base if specified
            if image_spec.from_registry:
                base = modal.Image.from_registry(
                    image_spec.from_registry,
                    add_python=image_spec.python_version
                )
            
            # Add apt packages
            if image_spec.apt_packages:
                base = base.apt_install(*image_spec.apt_packages)
            
            # Add pip packages
            if image_spec.pip_packages:
                base = base.pip_install(*image_spec.pip_packages)
            
            # Add environment variables
            if image_spec.env_vars:
                base = base.env(image_spec.env_vars)
            
            # Always add oven_core and workloads
            base = base.add_local_python_source("oven_core")
            base = base.add_local_python_source("workloads")
            
    except Exception:
        # Fall back to base image if anything goes wrong
        pass
    
    return base


# Default image for tasks without custom requirements
image = get_base_image()

@app.function(
    image=image, 
    volumes={"/mnt/artifacts": volume}, 
    timeout=3600
)
def modal_run_task(task_name: str, inputs: Dict[str, Any], run_id: str):
    """Execute a task on Modal infrastructure."""
    import importlib
    import datetime
    import traceback
    from pathlib import Path
    
    from oven_core.runtime import RunContext, RunResult, BaseTask
    from oven_core.decorators import get_task
    
    try:
        print(f"Starting task {task_name} for run {run_id}")
        run_dir = Path(f"/mnt/artifacts/runs/{run_id}")
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Load task - support both decorator and class-based
        print(f"Loading task: {task_name}")
        
        # Import workloads package to trigger decorator registrations
        try:
            import workloads
        except ImportError:
            pass
        
        # Try to import specific module
        try:
            module = importlib.import_module(f"workloads.{task_name}")
        except ImportError:
            module = None
        
        # Try decorator registry first
        task = get_task(task_name)
        
        if task is None and module is not None:
            # Fall back to class-based discovery
            task_cls = None
            for attr in dir(module):
                cls = getattr(module, attr)
                if (isinstance(cls, type) and 
                    any(b.__name__ == "BaseTask" for b in cls.__mro__) and 
                    cls.__name__ != "BaseTask"):
                    task_cls = cls
                    break
            
            if task_cls:
                task = task_cls()
                print(f"Task class {task_cls.__name__} found and instantiated")
        
        if task is None:
            raise ValueError(
                f"Task '{task_name}' not found. "
                f"Make sure it's defined as a @task decorated function or a BaseTask subclass."
            )
        
        if hasattr(task, '_fn'):
            print(f"Decorator-based task '{task_name}' loaded")
        
        # Setup paths on volume
        artifact_path = run_dir / "artifacts"
        artifact_path.mkdir(parents=True, exist_ok=True)
        
        context = RunContext(
            run_id=run_id,
            work_dir=run_dir,
            artifact_path=artifact_path,
            secrets={},
        )
        
        start_time = datetime.datetime.now()
        metadata = task.run(context, inputs)
        end_time = datetime.datetime.now()
        
        result = RunResult(
            run_id=run_id,
            status="succeeded",
            metadata={
                **metadata,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration": (end_time - start_time).total_seconds(),
                "backend": "modal",
            },
            artifact_path=f"runs/{run_id}/artifacts"
        )
        print(f"Task {task_name} completed successfully")
    except Exception as e:
        print(f"Task {task_name} failed: {e}")
        traceback.print_exc()
        result = RunResult(
            run_id=run_id,
            status="failed",
            error=str(e) + "\n" + traceback.format_exc()
        )
    
    # Save result to volume
    result_file = run_dir / "result.json"
    with open(result_file, "w") as f:
        f.write(result.model_dump_json(indent=2))
    
    # Ensure volume is committed
    volume.commit()
    return result.model_dump()

class ModalBackend(BaseBackend):
    """Modal execution backend for serverless GPU workloads."""
    
    def __init__(self, base_dir: Path = Path(".oven")):
        self.base_dir = base_dir
        self.runs_dir = self.base_dir / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def _save_local_metadata(self, run_id: str, data: Dict[str, Any]):
        metadata_file = self.runs_dir / f"{run_id}.json"
        with open(metadata_file, "w") as f:
            json.dump(data, f, indent=2)

    def _spawn_task(self, task_name: str, inputs: Dict[str, Any], run_id: str) -> str:
        """Spawn a single task and return the Modal call_id."""
        try:
            func = modal.Function.from_name("oven-workloads", "modal_run_task")
            handle = func.spawn(task_name, inputs, run_id)
            return handle.object_id
        except Exception:
            # Fallback to local app run if not deployed
            with app.run():
                handle = modal_run_task.spawn(task_name, inputs, run_id)
                return handle.object_id

    def submit(self, task_name: str, inputs: Dict[str, Any]) -> str:
        """Submit a single task for execution."""
        run_id = str(uuid.uuid4())[:8]
        
        # Save placeholder status locally
        self._save_local_metadata(run_id, {
            "run_id": run_id,
            "status": "pending",
            "backend": "modal",
            "task_name": task_name,
        })
        
        # Spawn the Modal function
        call_id = self._spawn_task(task_name, inputs, run_id)
        
        # Update local metadata with call_id
        self._save_local_metadata(run_id, {
            "run_id": run_id,
            "status": "running",
            "backend": "modal",
            "task_name": task_name,
            "modal_call_id": call_id,
        })
        
        return run_id

    def submit_batch(
        self,
        task_name: str,
        inputs_list: List[Dict[str, Any]],
        max_concurrency: Optional[int] = None,
    ) -> List[str]:
        """
        Submit multiple tasks in parallel on Modal.
        
        Modal handles parallelism natively, so we spawn all tasks at once.
        """
        run_ids = [str(uuid.uuid4())[:8] for _ in inputs_list]
        
        # Try to use the deployed function
        try:
            func = modal.Function.from_name("oven-workloads", "modal_run_task")
            
            for run_id, inputs in zip(run_ids, inputs_list):
                self._save_local_metadata(run_id, {
                    "run_id": run_id,
                    "status": "pending",
                    "backend": "modal",
                    "task_name": task_name,
                })
                
                handle = func.spawn(task_name, inputs, run_id)
                
                self._save_local_metadata(run_id, {
                    "run_id": run_id,
                    "status": "running",
                    "backend": "modal",
                    "task_name": task_name,
                    "modal_call_id": handle.object_id,
                })
                
        except Exception:
            # Fallback: run with local app context
            with app.run():
                for run_id, inputs in zip(run_ids, inputs_list):
                    self._save_local_metadata(run_id, {
                        "run_id": run_id,
                        "status": "pending",
                        "backend": "modal",
                        "task_name": task_name,
                    })
                    
                    handle = modal_run_task.spawn(task_name, inputs, run_id)
                    
                    self._save_local_metadata(run_id, {
                        "run_id": run_id,
                        "status": "running",
                        "backend": "modal",
                        "task_name": task_name,
                        "modal_call_id": handle.object_id,
                    })
        
        return run_ids

    def map(
        self,
        task_name: str,
        inputs_list: List[Dict[str, Any]],
        max_concurrency: Optional[int] = None,
    ) -> Iterator[Tuple[str, RunResult]]:
        """
        Map a task over multiple inputs using Modal's native parallelism.
        
        Yields results as they complete.
        """
        from modal.functions import FunctionCall
        
        run_ids = self.submit_batch(task_name, inputs_list, max_concurrency)
        
        # Collect all call IDs
        call_ids = []
        for run_id in run_ids:
            metadata_file = self.runs_dir / f"{run_id}.json"
            with open(metadata_file, "r") as f:
                data = json.load(f)
                call_ids.append((run_id, data.get("modal_call_id")))
        
        # Poll for results (in order for simplicity)
        for run_id, call_id in call_ids:
            if call_id:
                try:
                    call = FunctionCall.from_id(call_id)
                    result_data = call.get()  # Blocking wait
                    res = RunResult(**result_data)
                    self._save_local_metadata(run_id, res.model_dump())
                    yield run_id, res
                except Exception as e:
                    import traceback
                    error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                    res = RunResult(run_id=run_id, status="failed", error=error_msg)
                    self._save_local_metadata(run_id, res.model_dump())
                    yield run_id, res
            else:
                yield run_id, self.get_status(run_id)

    def get_status(self, run_id: str) -> RunResult:
        metadata_file = self.runs_dir / f"{run_id}.json"
        if not metadata_file.exists():
            raise ValueError(f"Run {run_id} not found")
            
        with open(metadata_file, "r") as f:
            local_data = json.load(f)
            
        if local_data["status"] in ["succeeded", "failed"]:
            return RunResult(**local_data)
            
        # Check Modal call status if still running
        call_id = local_data.get("modal_call_id")
        if call_id:
            from modal.functions import FunctionCall
            call = FunctionCall.from_id(call_id)
            try:
                # Non-blocking check
                result_data = call.get(timeout=0)
                if result_data:
                    res = RunResult(**result_data)
                    self._save_local_metadata(run_id, res.model_dump())
                    return res
            except TimeoutError:
                pass  # Still running
            except Exception as e:
                import traceback
                if hasattr(e, "exception"):
                    error_msg = f"RemoteError: {e.exception}\n{traceback.format_exc()}"
                else:
                    error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                res = RunResult(run_id=run_id, status="failed", error=error_msg)
                self._save_local_metadata(run_id, res.model_dump())
                return res
        
        return RunResult(**local_data)

    def fetch_artifacts(self, run_id: str, dest_path: Path) -> Path:
        result = self.get_status(run_id)
        if result.status != "succeeded":
            raise ValueError(f"Run {run_id} is not succeeded (status: {result.status})")
            
        remote_path = result.artifact_path
        if not remote_path:
            raise ValueError(f"No artifact path for run {run_id}")
            
        dest_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading artifacts from Modal volume: {remote_path} -> {dest_path}")
        
        for entry in volume.listdir(remote_path, recursive=True):
            if entry.type == modal.volume.FileEntryType.FILE:
                local_rel_path = Path(entry.path).relative_to(remote_path)
                local_file = dest_path / local_rel_path
                local_file.parent.mkdir(parents=True, exist_ok=True)
                
                with open(local_file, "wb") as f:
                    for chunk in volume.read_file(entry.path):
                        f.write(chunk)
        
        return dest_path

