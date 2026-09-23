"""Хранилище диалогов и лидов (SQLite). Диалоги переживают перезапуск бота."""
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS dialogs (
    user_id     INTEGER PRIMARY KEY,
    username    TEXT,
    first_name  TEXT,
    stage       TEXT NOT NULL DEFAULT 'qualifying',  -- qualifying | contact | done
    fields      TEXT NOT NULL DEFAULT '{}',
    history     TEXT NOT NULL DEFAULT '[]',
    pending     TEXT,
    course_id   TEXT,
    updated_at  TEXT NOT NULL,
    nudged      INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS leads (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      INTEGER NOT NULL,
    name         TEXT,
    phone        TEXT,
    course_id    TEXT,
    points       INTEGER NOT NULL,
    temperature  TEXT NOT NULL,
    crm_id       INTEGER,
    created_at   TEXT NOT NULL
);
"""
HISTORY_LIMIT = 24


@dataclass
class Dialog:
    user_id: int
    username: str | None
    first_name: str
    stage: str
    fields: dict
    history: list[dict]
    pending: str | None
    course_id: str | None
    updated_at: datetime
    nudged: bool

    def say(self, role: str, text: str) -> None:
        self.history.append({"role": role, "content": text})
        del self.history[:-HISTORY_LIMIT]


@dataclass
class Lead:
    id: int
    name: str | None
    phone: str | None
    course_id: str
    points: int
    temperature: str
    crm_id: int | None
    created_at: datetime


class Store:
    def __init__(self, path: Path):
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    @staticmethod
    def _dialog(r: sqlite3.Row) -> Dialog:
        return Dialog(
            user_id=r["user_id"], username=r["username"], first_name=r["first_name"] or "",
            stage=r["stage"], fields=json.loads(r["fields"]), history=json.loads(r["history"]),
            pending=r["pending"], course_id=r["course_id"],
            updated_at=datetime.fromisoformat(r["updated_at"]), nudged=bool(r["nudged"]),
        )

    def get(self, user_id: int) -> Dialog | None:
        r = self.conn.execute("SELECT * FROM dialogs WHERE user_id = ?", (user_id,)).fetchone()
        return self._dialog(r) if r else None

    def new(self, user_id: int, username: str | None, first_name: str, now: datetime) -> Dialog:
        d = Dialog(user_id, username, first_name, "qualifying", {}, [], None, None, now, False)
        self.save(d)
        return d

    def save(self, d: Dialog) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO dialogs (user_id, username, first_name, stage, fields, history,"
            " pending, course_id, updated_at, nudged) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (d.user_id, d.username, d.first_name, d.stage, json.dumps(d.fields, ensure_ascii=False),
             json.dumps(d.history, ensure_ascii=False), d.pending, d.course_id,
             d.updated_at.isoformat(), int(d.nudged)),
        )
        self.conn.commit()

    def stale(self, before: datetime) -> list[Dialog]:
        """Незавершённые диалоги, где клиент замолчал и мы ещё не напоминали."""
        rows = self.conn.execute(
            "SELECT * FROM dialogs WHERE stage != 'done' AND nudged = 0 AND updated_at < ?",
            (before.isoformat(),),
        ).fetchall()
        dialogs = [self._dialog(r) for r in rows]
        return [d for d in dialogs if any(m["role"] == "user" for m in d.history)]

    def add_lead(self, *, user_id: int, name: str | None, phone: str | None, course_id: str,
                 points: int, temperature: str, crm_id: int | None, now: datetime) -> None:
        self.conn.execute(
            "INSERT INTO leads (user_id, name, phone, course_id, points, temperature, crm_id, created_at)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (user_id, name, phone, course_id, points, temperature, crm_id, now.isoformat()),
        )
        self.conn.commit()

    def recent_leads(self, limit: int = 10) -> list[Lead]:
        rows = self.conn.execute("SELECT * FROM leads ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [Lead(r["id"], r["name"], r["phone"], r["course_id"], r["points"], r["temperature"],
                     r["crm_id"], datetime.fromisoformat(r["created_at"])) for r in rows]
