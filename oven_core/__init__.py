"""
Oven Core - Serverless workload orchestration framework.

This package provides:
- Task definitions (class-based and decorator-based)
- Runtime primitives (RunContext, RunResult, WorkloadSpec)
- Backend implementations (Local, Modal)
"""

from oven_core.runtime import (
    BaseTask,
    BaseBackend,
    WorkloadSpec,
    RunContext,
    RunResult,
    BatchResult,
    RetryPolicy,
)

from oven_core.decorators import (
    task,
    image,
    ImageSpec,
    get_task,
    list_tasks,
    FunctionTask,
)

__all__ = [
    # Runtime
    "BaseTask",
    "BaseBackend",
    "WorkloadSpec",
    "RunContext",
    "RunResult",
    "BatchResult",
    "RetryPolicy",
    # Decorators
    "task",
    "image",
    "ImageSpec",
    "get_task",
    "list_tasks",
    "FunctionTask",
]

