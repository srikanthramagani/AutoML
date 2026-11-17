"""
Config package.
Exports the Settings dataclass and the module-level default_settings instance.
"""
from .settings import Settings, default_settings

__all__ = ["Settings", "default_settings"]
