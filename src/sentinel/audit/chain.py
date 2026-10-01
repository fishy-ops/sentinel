import hashlib
import json
from datetime import UTC, datetime

from sqlalchemy import Engine, select, text
from sqlalchemy.orm import Session

from sentinel.store.models import AuditLog


def _digest(row: AuditLog) -> str:
    fields = {
        "sequence": row.sequence,
        "timestamp": row.timestamp,
        "request_id": row.request_id,
        "key_prefix": row.key_prefix,
        "action": row.action,
        "resource_id": row.resource_id,
        "outcome": row.outcome,
        "status_code": row.status_code,
        "prev_hash": row.prev_hash,
    }
    data = json.dumps(fields, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(data.encode()).hexdigest()


def append(
    engine: Engine,
    request_id: str,
    key_prefix: str | None,
    action: str,
    resource_id: str | None,
    outcome: str,
    status_code: int,
) -> None:
    with engine.connect() as connection:
        connection.execute(text("BEGIN IMMEDIATE"))
        try:
            with Session(bind=connection) as session:
                previous = session.scalar(
                    select(AuditLog).order_by(AuditLog.sequence.desc()).limit(1)
                )
                row = AuditLog(
                    sequence=1 if previous is None else previous.sequence + 1,
                    timestamp=datetime.now(UTC).isoformat(),
                    request_id=request_id,
                    key_prefix=key_prefix,
                    action=action,
                    resource_id=resource_id,
                    outcome=outcome,
                    status_code=status_code,
                    prev_hash="0" * 64 if previous is None else previous.entry_hash,
                    entry_hash="",
                )
                row.entry_hash = _digest(row)
                session.add(row)
                session.flush()
            connection.commit()
        except Exception:
            connection.rollback()
            raise


def verify(engine: Engine) -> tuple[bool, int | None]:
    with Session(engine) as session:
        rows = session.scalars(select(AuditLog).order_by(AuditLog.sequence)).all()
    expected_sequence = 1
    previous_hash = "0" * 64
    for row in rows:
        if row.sequence != expected_sequence:
            return False, expected_sequence
        if row.prev_hash != previous_hash or row.entry_hash != _digest(row):
            return False, row.sequence
        expected_sequence += 1
        previous_hash = row.entry_hash
    return True, None
