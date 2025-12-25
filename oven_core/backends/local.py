import json
import uuid
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Iterator, Tuple
import importlib
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from oven_core.runtime import (
    BaseBackend, BaseTask, RunContext, RunResult, WorkloadSpec, BatchResult
)


class LocalBackend(BaseBackend):
    """Local execution backend for development and testing."""
    
    def __init__(self, base_dir: Path = Path(".oven"), max_workers: int = 4):
        self.base_dir = base_dir
        self.runs_dir = self.base_dir / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers

    def _get_task(self, task_name: str) -> BaseTask:
        """Load a task by name, supporting both class-based and decorator-based tasks."""
        from oven_core.decorators import get_task
        
        # First, import the workloads package to trigger all decorator registrations
        try:
            import workloads
        except ImportError:
            pass
        
        # Try to import the specific module (for class-based tasks or task-specific files)
        try:
            importlib.import_module(f"workloads.{task_name}")
        except ImportError:
            pass  # Task might be registered via decorator in another module
        
        # Check decorator registry first
        fn_task = get_task(task_name)
        if fn_task is not None:
            return fn_task
        
        # Fall back to class-based task discovery
        try:
            return self._get_task_class(task_name)()
        except ValueError:
            raise ValueError(
                f"Task '{task_name}' not found. "
                f"Make sure it's defined as a @task decorated function or a BaseTask subclass."
            )
    
    def _get_task_class(self, task_name: str) -> Type[BaseTask]:
        """Load a class-based task (legacy support)."""
        try:
            module = importlib.import_module(f"workloads.{task_name}")
            # Look for a class that inherits from BaseTask
            for attr in dir(module):
                cls = getattr(module, attr)
                if isinstance(cls, type) and issubclass(cls, BaseTask) and cls is not BaseTask:
                    return cls
            raise ValueError(f"No BaseTask subclass found in workloads.{task_name}")
        except ImportError as e:
            raise ValueError(f"Could not import workload {task_name}: {e}")

    def _save_run_result(self, result: RunResult):
        run_file = self.runs_dir / f"{result.run_id}.json"
        with open(run_file, "w") as f:
            f.write(result.model_dump_json(indent=2))

    def _execute_task(self, task_name: str, inputs: Dict[str, Any], run_id: Optional[str] = None) -> str:
        """Internal method to execute a single task."""
        if run_id is None:
            run_id = str(uuid.uuid4())[:8]
        
        task = self._get_task(task_name)
        
        run_dir = self.runs_dir / run_id
        artifact_path = run_dir / "artifacts"
        artifact_path.mkdir(parents=True, exist_ok=True)

        context = RunContext(
            run_id=run_id,
            work_dir=run_dir,
            artifact_path=artifact_path,
            secrets={},  # Local secrets could be loaded from env
        )

        result = RunResult(run_id=run_id, status="running")
        self._save_run_result(result)

        try:
            start_time = datetime.datetime.now()
            metadata = task.run(context, inputs)
            end_time = datetime.datetime.now()
            
            result.status = "succeeded"
            result.metadata = {
                **metadata,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration": (end_time - start_time).total_seconds(),
                "backend": "local",
            }
            result.artifact_path = str(artifact_path)
        except Exception as e:
            result.status = "failed"
            result.error = str(e)
            import traceback
            print(traceback.format_exc())

        self._save_run_result(result)
        return run_id

    def submit(self, task_name: str, inputs: Dict[str, Any]) -> str:
        """Submit a single task for execution."""
        return self._execute_task(task_name, inputs)

    def submit_batch(
        self,
        task_name: str,
        inputs_list: List[Dict[str, Any]],
        max_concurrency: Optional[int] = None,
    ) -> List[str]:
        """
        Submit multiple tasks in parallel using ThreadPoolExecutor.
        
        Args:
            task_name: Name of the task to run
            inputs_list: List of input dictionaries
            max_concurrency: Maximum concurrent tasks (defaults to self.max_workers)
            
        Returns:
            List of run_ids in the same order as inputs_list
        """
        workers = max_concurrency or self.max_workers
        run_ids = [str(uuid.uuid4())[:8] for _ in inputs_list]
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self._execute_task, task_name, inputs, run_id): idx
                for idx, (inputs, run_id) in enumerate(zip(inputs_list, run_ids))
            }
            
            # Wait for all to complete
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    # Error is already saved in _execute_task
                    pass
        
        return run_ids

    def map(
        self,
        task_name: str,
        inputs_list: List[Dict[str, Any]],
        max_concurrency: Optional[int] = None,
    ) -> Iterator[Tuple[str, RunResult]]:
        """
        Map a task over multiple inputs and yield results as they complete.
        
        This implementation runs tasks in parallel and yields results
        as each task finishes (not necessarily in order).
        """
        workers = max_concurrency or self.max_workers
        run_ids = [str(uuid.uuid4())[:8] for _ in inputs_list]
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self._execute_task, task_name, inputs, run_id): run_id
                for inputs, run_id in zip(inputs_list, run_ids)
            }
            
            for future in as_completed(futures):
                run_id = futures[future]
                yield run_id, self.get_status(run_id)

    def get_status(self, run_id: str) -> RunResult:
        run_file = self.runs_dir / f"{run_id}.json"
        if not run_file.exists():
            raise ValueError(f"Run {run_id} not found")
        with open(run_file, "r") as f:
            data = json.load(f)
            return RunResult(**data)

    def fetch_artifacts(self, run_id: str, dest_path: Path) -> Path:
        result = self.get_status(run_id)
        if not result.artifact_path:
            raise ValueError(f"No artifacts for run {run_id}")
        
        src_path = Path(result.artifact_path)
        if not src_path.exists():
            raise ValueError(f"Artifact path {src_path} does not exist")
        
        # In local backend, just return the path
        return src_path

