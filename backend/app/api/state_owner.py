"""Thread-local owner context for persistent datasets and jobs."""
from contextlib import contextmanager
from threading import local

_context = local()


def set_owner_id(owner_id: str) -> None:
    _context.owner_id = owner_id


def get_owner_id() -> str:
    return getattr(_context, "owner_id", "local")
