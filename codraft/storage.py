import json
import sqlite3
import datetime
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from codraft.types import DraftState, Section, PatchOperation

class BaseStorage(ABC):
    @abstractmethod
    def init_session(self, session_id: str, initial_text: str = "") -> DraftState:
        pass

    @abstractmethod
    def get_state(self, session_id: str) -> Optional[DraftState]:
        pass

    @abstractmethod
    def save_state(self, state: DraftState, patch: Optional[PatchOperation] = None) -> int:
        pass

    @abstractmethod
    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        pass


class SQLiteStorage(BaseStorage):
    def __init__(self, db_path: str = "codraft_workspace.db", history_file: str = "draft_history.jsonl"):
        self.db_path = db_path
        self.history_file = history_file
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    version INTEGER NOT NULL DEFAULT 0,
                    full_text TEXT NOT NULL DEFAULT '',
                    last_editor TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sections (
                    session_id TEXT NOT NULL,
                    section_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    last_editor TEXT NOT NULL,
                    last_version INTEGER NOT NULL,
                    PRIMARY KEY (session_id, section_id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS patch_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    editor TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    section_id TEXT NOT NULL,
                    reason TEXT,
                    content TEXT,
                    confidence REAL,
                    timestamp TEXT NOT NULL
                )
            """)
            conn.commit()

    def init_session(self, session_id: str, initial_text: str = "") -> DraftState:
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self._get_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO sessions (session_id, version, full_text, last_editor, updated_at) "
                "VALUES (?, 0, ?, '', ?)",
                (session_id, initial_text, now)
            )
            conn.execute("DELETE FROM sections WHERE session_id = ?", (session_id,))
            conn.commit()
        return DraftState(
            session_id=session_id,
            version=0,
            full_text=initial_text,
            sections={},
            last_editor="",
            timestamp=now
        )

    def get_state(self, session_id: str) -> Optional[DraftState]:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT session_id, version, full_text, last_editor, updated_at FROM sessions WHERE session_id = ?",
                (session_id,)
            ).fetchone()
            if not row:
                return None

            sec_rows = conn.execute(
                "SELECT section_id, title, content, last_editor, last_version FROM sections WHERE session_id = ?",
                (session_id,)
            ).fetchall()
            
            sections = {
                r["section_id"]: Section(
                    id=r["section_id"],
                    title=r["title"],
                    content=r["content"],
                    last_editor=r["last_editor"],
                    last_version=r["last_version"]
                ) for r in sec_rows
            }

            return DraftState(
                session_id=row["session_id"],
                version=row["version"],
                full_text=row["full_text"],
                sections=sections,
                last_editor=row["last_editor"],
                timestamp=row["updated_at"]
            )

    def save_state(self, state: DraftState, patch: Optional[PatchOperation] = None) -> int:
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        state.version += 1
        state.timestamp = now

        with self._get_conn() as conn:
            conn.execute(
                "UPDATE sessions SET version = ?, full_text = ?, last_editor = ?, updated_at = ? "
                "WHERE session_id = ?",
                (state.version, state.full_text, state.last_editor, now, state.session_id)
            )
            for sec in state.sections.values():
                conn.execute(
                    "INSERT OR REPLACE INTO sections (session_id, section_id, title, content, last_editor, last_version) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (state.session_id, sec.id, sec.title, sec.content, sec.last_editor, state.version)
                )

            if patch:
                conn.execute(
                    "INSERT INTO patch_log (session_id, version, editor, operation, section_id, reason, content, confidence, timestamp) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (state.session_id, state.version, patch.model or state.last_editor, patch.operation,
                     patch.section_id, patch.reason, patch.content, patch.confidence or 0.8, now)
                )
            conn.commit()

        if self.history_file and patch:
            log_record = {
                "session_id": state.session_id,
                "version": state.version,
                "timestamp": now,
                "editor": patch.model or state.last_editor,
                "operation": patch.operation,
                "section_id": patch.section_id,
                "reason": patch.reason,
                "content_len": len(patch.content),
                "confidence": patch.confidence
            }
            try:
                with open(self.history_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_record) + "\n")
            except Exception:
                pass

        return state.version

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT version, editor, operation, section_id, reason, content, confidence, timestamp "
                "FROM patch_log WHERE session_id = ? ORDER BY version ASC",
                (session_id,)
            ).fetchall()
            return [dict(r) for r in rows]


class InMemoryStorage(BaseStorage):
    def __init__(self):
        self.sessions: Dict[str, DraftState] = {}
        self.logs: Dict[str, List[Dict[str, Any]]] = {}

    def init_session(self, session_id: str, initial_text: str = "") -> DraftState:
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        state = DraftState(
            session_id=session_id,
            version=0,
            full_text=initial_text,
            sections={},
            last_editor="",
            timestamp=now
        )
        self.sessions[session_id] = state
        self.logs[session_id] = []
        return state

    def get_state(self, session_id: str) -> Optional[DraftState]:
        return self.sessions.get(session_id)

    def save_state(self, state: DraftState, patch: Optional[PatchOperation] = None) -> int:
        state.version += 1
        state.timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.sessions[state.session_id] = state
        if patch:
            self.logs[state.session_id].append({
                "version": state.version,
                "editor": patch.model or state.last_editor,
                "operation": patch.operation,
                "section_id": patch.section_id,
                "reason": patch.reason,
                "content": patch.content,
                "confidence": patch.confidence,
                "timestamp": state.timestamp
            })
        return state.version

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        return self.logs.get(session_id, [])
