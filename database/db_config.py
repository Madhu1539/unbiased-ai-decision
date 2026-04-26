"""
db_config.py
------------
SQLAlchemy + SQLite configuration.
Used for persisting session logs and run history.
"""
import os

from sqlalchemy import Column, DateTime, Integer, JSON, String, create_engine, func
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# ── Database path ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, 'unbiased_ai.db')}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


# ── ORM Models ─────────────────────────────────────────────────────────
class RunRecord(Base):
    """Stores a summary of each ML training run."""
    __tablename__ = "run_records"

    id = Column(Integer, primary_key=True, index=True)
    dataset_name = Column(String, nullable=True)
    model_name = Column(String, nullable=True)
    task_type = Column(String, nullable=True)
    metrics = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class PreprocessingLog(Base):
    """Stores preprocessing step logs per run."""
    __tablename__ = "preprocessing_logs"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, nullable=True)
    log_entries = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency: yields a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
