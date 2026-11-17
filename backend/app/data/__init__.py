"""
Data package.
Contains modules for loading CSVs, profiling datasets, and detecting the ML task type.
"""
from .loader import DataLoader
from .profiler import DataProfiler
from .task_detector import TaskDetector

__all__ = ["DataLoader", "DataProfiler", "TaskDetector"]
