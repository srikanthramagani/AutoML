"""Compatibility state backed by SQLite persistent metadata."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from collections import UserDict

from ..database.database import SessionLocal, engine
from ..database.models import Base
from ..database.crud import get_record, save_record
from .state_owner import get_owner_id

PROJECT_ROOT = Path(__file__).resolve().parents[3]
Base.metadata.create_all(bind=engine)


class PersistentPayload(dict):
    def __init__(self, state: "PersistentState", record_id: str, payload: Dict[str, Any]) -> None:
        super().__init__(payload)
        self._state = state
        self._record_id = record_id

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, value)
        self._persist()

    def update(self, *args: Any, **kwargs: Any) -> None:
        super().update(*args, **kwargs)
        self._persist()

    def _persist(self) -> None:
        self._state[self._record_id] = dict(self)


class PersistentState(UserDict):
    def __init__(self, record_type: str, owner_id: str = "default") -> None:
        super().__init__()
        self.record_type = record_type
        self.owner_id = owner_id
        self._load()

    def _effective_owner(self) -> str:
        owner_id = get_owner_id()
        return owner_id if owner_id else self.owner_id

    def _load(self) -> None:
        # Jobs and datasets are restored lazily because payloads are keyed by ID.
        self.data.clear()

    def __getitem__(self, key: str) -> Dict[str, Any]:
        value = self.data.get(key)
        if value is not None:
            return value
        session = SessionLocal()
        try:
            value = get_record(session, self._effective_owner(), self.record_type, str(key))
        finally:
            session.close()
        if value is None:
            raise KeyError(key)
        persistent_value = PersistentPayload(self, str(key), value)
        self.data[key] = persistent_value
        return persistent_value

    def __setitem__(self, key: str, value: Dict[str, Any]) -> None:
        persistent_value = PersistentPayload(self, str(key), value)
        self.data[key] = persistent_value
        session = SessionLocal()
        try:
            save_record(session, self._effective_owner(), self.record_type, str(key), value)
        finally:
            session.close()

    def update(self, *args: Any, **kwargs: Any) -> None:
        payload = dict(*args, **kwargs)
        if "job_id" in payload:
            record_id = str(payload["job_id"])
        elif "id" in payload:
            record_id = str(payload["id"])
        else:
            return super().update(payload)
        merged = dict(self.data.get(record_id, {}))
        merged.update(payload)
        self[record_id] = merged

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: object) -> bool:
        try:
            self[str(key)]
            return True
        except (KeyError, TypeError):
            return False


DATASETS = PersistentState("dataset")
JOBS = PersistentState("job")
