"""
Workloads package - contains all task definitions.

Import all task modules here to register decorator-based tasks.
"""

# Import all task modules to register them in the decorator registry
from workloads import demo_task

# List all available task modules
__all__ = ["demo_task"]

