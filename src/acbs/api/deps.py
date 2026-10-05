from __future__ import annotations

from acbs.storage.models import init_db, make_session_factory

SessionLocal = make_session_factory()
init_db()


def get_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
