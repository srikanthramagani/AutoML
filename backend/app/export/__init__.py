"""
Export package.
Handles serialization of the final trained pipeline to a .pkl file.
"""
from .model_exporter import ModelExporter

__all__ = ["ModelExporter"]
