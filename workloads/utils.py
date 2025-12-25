"""
Shared utilities for workloads.

This module contains common functions and classes that can be imported
by multiple tasks. Example usage:

    from workloads.utils import save_json, Timer, DataProcessor
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List
from dataclasses import dataclass


def save_json(path: Path, data: Dict[str, Any], indent: int = 2) -> None:
    """Save data as JSON to a file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON data from a file."""
    with open(path) as f:
        return json.load(f)


class Timer:
    """Simple context manager for timing code blocks."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.elapsed = 0.0
    
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start
        if self.name:
            print(f"[{self.name}] took {self.elapsed:.3f}s")


@dataclass
class DataProcessor:
    """
    A reusable data processor for batch operations.
    
    Example:
        processor = DataProcessor(batch_size=100)
        for batch in processor.batches(items):
            process(batch)
    """
    batch_size: int = 32
    
    def batches(self, items: List[Any]):
        """Yield batches of items."""
        for i in range(0, len(items), self.batch_size):
            yield items[i:i + self.batch_size]
    
    def process_all(self, items: List[Any], fn) -> List[Any]:
        """Apply a function to all items in batches."""
        results = []
        for batch in self.batches(items):
            results.extend([fn(item) for item in batch])
        return results


# Common configuration defaults
DEFAULT_CONFIG = {
    "seed": 42,
    "verbose": True,
    "max_retries": 3,
}

