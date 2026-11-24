"""Persistent metadata models for datasets and training jobs."""
from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class PersistentRecord(Base):
    __tablename__ = "persistent_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    owner_id = Column(String(128), nullable=False, index=True)
    record_type = Column(String(32), nullable=False, index=True)
    record_id = Column(String(128), nullable=False, index=True)
    payload_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
