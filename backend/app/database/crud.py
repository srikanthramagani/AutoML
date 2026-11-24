"""Persistent record CRUD helpers."""
import json
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from .models import PersistentRecord


def save_record(
    session: Session,
    owner_id: str,
    record_type: str,
    record_id: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    record = (
        session.query(PersistentRecord)
        .filter(
            PersistentRecord.owner_id == owner_id,
            PersistentRecord.record_type == record_type,
            PersistentRecord.record_id == record_id,
        )
        .first()
    )
    payload_json = json.dumps(payload, ensure_ascii=False, default=str)
    if record is None:
        record = PersistentRecord(
            owner_id=owner_id,
            record_type=record_type,
            record_id=record_id,
            payload_json=payload_json,
        )
        session.add(record)
    else:
        record.payload_json = payload_json
    session.commit()
    return payload


def get_record(
    session: Session,
    owner_id: str,
    record_type: str,
    record_id: str,
) -> Optional[Dict[str, Any]]:
    record = (
        session.query(PersistentRecord)
        .filter(
            PersistentRecord.owner_id == owner_id,
            PersistentRecord.record_type == record_type,
            PersistentRecord.record_id == record_id,
        )
        .first()
    )
    return json.loads(record.payload_json) if record is not None else None
