
from __future__ import annotations
import os, time
from contextlib import contextmanager
from sqlalchemy import create_engine, Integer, String, Boolean, Float, Column, select, func
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session

# DATABASE_URL examples:
# - Codespaces (compose): postgresql+psycopg://postgres:postgres@db:5432/tasks
# - Local without compose: postgresql+psycopg://postgres:postgres@localhost:5432/tasks
# - Managed (Neon/Supabase): postgresql+psycopg://user:pass@host/db?sslmode=require
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost:5432/tasks")

engine = create_engine(DATABASE_URL, future=True, pool_pre_ping=True)
SessionLocal = scoped_session(sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True))
Base = declarative_base()

class Task(Base):
    __tablename__ = "tasks"
    id = Column(Integer, primary_key=True)
    description = Column(String, nullable=False)
    completed = Column(Boolean, nullable=False, default=False)
    created_at = Column(Float, nullable=False)      # epoch seconds (float) to match API
    due_at = Column(Float, nullable=True)           # epoch seconds (float) or None
    priority = Column(Integer, nullable=False, default=2)
    order = Column(Integer, nullable=False, default=0)

def init_db() -> None:
    Base.metadata.create_all(bind=engine)

@contextmanager
def session_scope():
    s = SessionLocal()
    try:
        yield s
        s.commit()
    except:
        s.rollback()
        raise
    finally:
        s.close()

class TaskRepo:
    """DB-backed repository with same surface as the prior in-memory store."""
    def add(self, description: str, *, due_at: float | None = None, priority: int = 2) -> Task:
        with session_scope() as s:
            max_order = s.scalar(select(func.max(Task.order))) or 0
            t = Task(
                description=description.strip(),
                completed=False,
                created_at=time.time(),
                due_at=due_at,
                priority=max(1, min(3, int(priority or 2))),
                order=int(max_order) + 1,
            )
            s.add(t)
            s.flush()
            s.refresh(t)
            return t

    def all(self) -> list[Task]:
        with session_scope() as s:
            return list(s.execute(select(Task).order_by(Task.order, Task.created_at)).scalars().all())

    def complete(self, task_id: int) -> Task | None:
        with session_scope() as s:
            t = s.get(Task, task_id)
            if not t:
                return None
            t.completed = True
            s.flush(); s.refresh(t)
            return t

    def update(self, task_id: int, *, description=None, due_at=None, priority=None, order=None) -> Task | None:
        with session_scope() as s:
            t = s.get(Task, task_id)
            if not t:
                return None
            if description is not None: t.description = description
            if due_at is not None: t.due_at = due_at
            if priority is not None: t.priority = max(1, min(3, int(priority)))
            if order is not None: t.order = int(order)
            s.flush(); s.refresh(t)
            return t

    def delete(self, task_id: int) -> bool:
        with session_scope() as s:
            t = s.get(Task, task_id)
            if not t: return False
            s.delete(t)
            return True

    def reorder(self, id_list: list[int]):
        with session_scope() as s:
            cur = 1
            seen: set[int] = set()
            for tid in id_list:
                t = s.get(Task, tid)
                if t and tid not in seen:
                    t.order = cur; cur += 1; seen.add(tid)
            remaining = s.execute(select(Task).order_by(Task.order, Task.created_at)).scalars().all()
            for t in remaining:
                if t.id not in seen:
                    t.order = cur; cur += 1

    def complete_all(self) -> int:
        with session_scope() as s:
            q = s.execute(select(Task).where(Task.completed.is_(False))).scalars().all()
            for t in q: t.completed = True
            return len(q)

    def delete_all(self) -> int:
        with session_scope() as s:
            q = s.execute(select(Task)).scalars().all()
            n = len(q)
            for t in q: s.delete(t)
            return n

    # Helpers for description-matching bulk ops
    def _match_ids_by_description(self, text: str) -> list[int]:
        q = (text or "").strip().lower()
        if not q: return []
        with session_scope() as s:
            items = s.execute(select(Task).order_by(Task.order, Task.created_at)).scalars().all()
            return [t.id for t in items if q in t.description.lower()]

    def complete_by_description(self, text: str) -> int:
        ids = self._match_ids_by_description(text)
        with session_scope() as s:
            n = 0
            for tid in ids:
                t = s.get(Task, tid)
                if t and not t.completed:
                    t.completed = True
                    n += 1
            return n

    def delete_by_description(self, text: str) -> int:
        ids = self._match_ids_by_description(text)
        with session_scope() as s:
            n = 0
            for tid in ids:
                t = s.get(Task, tid)
                if t is not None:
                    s.delete(t); n += 1
            return n
