"""
Example task that imports from a shared module.

This demonstrates how to structure workloads with shared utilities.

Usage:
    python cli.py run batch_job --backend local -i items='[1,2,3,4,5]'
    python cli.py run batch_job --backend modal -i batch_size=10
"""

from typing import Any, Dict, List

from oven_core import task, RunContext, ImageSpec

# Import shared utilities from the same package
from workloads.utils import Timer, DataProcessor, save_json, DEFAULT_CONFIG


@task(cpu=0.25, memory=128)
def batch_job(
    context: RunContext,
    items: List[int] = None,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """
    Process items in batches using shared utilities.
    
    This task demonstrates importing from workloads/utils.py.
    """
    if items is None:
        items = list(range(100))
    
    processor = DataProcessor(batch_size=batch_size)
    
    # Use the Timer utility
    with Timer("batch processing"):
        # Square each number
        results = processor.process_all(items, lambda x: x ** 2)
    
    output = {
        "input_count": len(items),
        "output_count": len(results),
        "batch_size": batch_size,
        "sum": sum(results),
        "config": DEFAULT_CONFIG,
    }
    
    # Use save_json utility
    save_json(context.artifact_path / "results.json", output)
    
    return output

