"""
Workloads package - contains all task definitions.

Import all task modules here to register decorator-based tasks.
"""

# Import all task modules to register them in the decorator registry
from workloads import demo_task
from workloads import batch_job

# List all available task modules
__all__ = ["demo_task", "batch_job", "utils"]

