from typing import Any, Dict, List, Optional, Literal, Iterator, Tuple
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
from pathlib import Path


# GPU type literals for type safety
GPU_TYPE = Literal[
    "T4", "L4", "A10G", "L40S", "A100", "A100-80GB", "H100", "H200"
]


class RetryPolicy(BaseModel):
    """Configuration for automatic retries on failure."""
    max_retries: int = 0
    initial_delay: float = 1.0
    backoff_multiplier: float = 2.0


class WorkloadSpec(BaseModel):
    """Specification for a workload's resource requirements."""
    name: str
    cpu: float = 0.1
    memory: int = 128  # MB
    gpu: Optional[str] = None  # GPU type string like "A10G", "H100:2"
    timeout: int = 600  # seconds
    retries: RetryPolicy = Field(default_factory=RetryPolicy)
    scaledown_window: int = 300  # seconds to keep container alive


class RunContext(BaseModel):
    """Context passed to task execution."""
    run_id: str
    work_dir: Path
    artifact_path: Path
    secrets: Dict[str, str] = Field(default_factory=dict)
    
    model_config = {"arbitrary_types_allowed": True}


class RunResult(BaseModel):
    """Result of a task execution."""
    run_id: str
    status: str  # "pending", "running", "succeeded", "failed"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    artifact_path: Optional[str] = None
    error: Optional[str] = None


class BatchResult(BaseModel):
    """Result of a batch/map operation."""
    run_ids: List[str]
    results: List[RunResult] = Field(default_factory=list)
    
    def succeeded(self) -> List[RunResult]:
        """Return only succeeded results."""
        return [r for r in self.results if r.status == "succeeded"]
    
    def failed(self) -> List[RunResult]:
        """Return only failed results."""
        return [r for r in self.results if r.status == "failed"]


class BaseTask(ABC):
    """Abstract base class for task definitions."""
    
    @property
    @abstractmethod
    def spec(self) -> WorkloadSpec:
        """Return the workload specification."""
        pass

    @abstractmethod
    def run(self, context: RunContext, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        The main computation logic.
        
        Args:
            context: Execution context with paths and secrets
            inputs: Dictionary of input parameters
            
        Returns:
            Dictionary of output metadata to be stored in RunResult
        """
        pass


class BaseBackend(ABC):
    """Abstract base class for execution backends."""
    
    @abstractmethod
    def submit(self, task_name: str, inputs: Dict[str, Any]) -> str:
        """Submit a task for execution and return a run_id."""
        pass

    @abstractmethod
    def get_status(self, run_id: str) -> RunResult:
        """Get the status of a run."""
        pass

    @abstractmethod
    def fetch_artifacts(self, run_id: str, dest_path: Path) -> Path:
        """Fetch artifacts for a run to a destination path."""
        pass
    
    # Optional: Batch/map operations (default implementation)
    def submit_batch(
        self, 
        task_name: str, 
        inputs_list: List[Dict[str, Any]],
        max_concurrency: Optional[int] = None,
    ) -> List[str]:
        """
        Submit multiple tasks in parallel.
        
        Args:
            task_name: Name of the task to run
            inputs_list: List of input dictionaries, one per task
            max_concurrency: Maximum concurrent tasks (None = unlimited)
            
        Returns:
            List of run_ids
        """
        # Default implementation: sequential submission
        return [self.submit(task_name, inputs) for inputs in inputs_list]
    
    def map(
        self,
        task_name: str,
        inputs_list: List[Dict[str, Any]],
        max_concurrency: Optional[int] = None,
    ) -> Iterator[Tuple[str, RunResult]]:
        """
        Map a task over multiple inputs and yield results as they complete.
        
        Args:
            task_name: Name of the task to run
            inputs_list: List of input dictionaries
            max_concurrency: Maximum concurrent tasks
            
        Yields:
            Tuples of (run_id, RunResult) as each task completes
        """
        run_ids = self.submit_batch(task_name, inputs_list, max_concurrency)
        for run_id in run_ids:
            yield run_id, self.get_status(run_id)
    
    def gather(
        self,
        task_name: str,
        inputs_list: List[Dict[str, Any]],
        max_concurrency: Optional[int] = None,
    ) -> BatchResult:
        """
        Run multiple tasks and gather all results.
        
        Args:
            task_name: Name of the task to run
            inputs_list: List of input dictionaries
            max_concurrency: Maximum concurrent tasks
            
        Returns:
            BatchResult containing all run results
        """
        run_ids = self.submit_batch(task_name, inputs_list, max_concurrency)
        results = [self.get_status(run_id) for run_id in run_ids]
        return BatchResult(run_ids=run_ids, results=results)

