"""SQLite persistence. Session IDs are bearer secrets; keep them private."""
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, path):
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, state TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS leads (
                    id TEXT PRIMARY KEY, session_id TEXT UNIQUE NOT NULL,
                    full_name TEXT NOT NULL, phone TEXT NOT NULL, company TEXT NOT NULL,
                    summary TEXT NOT NULL, created_at TEXT NOT NULL);
            ''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def create(self):
        state = dict(id=uuid.uuid4().hex, history=[], stage=None, fields={}, lead_id=None)
        self.save(state)
        return state

    def get(self, session_id):
        with self.connect() as db:
            row = db.execute('SELECT state FROM sessions WHERE id=?', (session_id,)).fetchone()
        if row is None:
            raise KeyError('Диалог не найден. Создайте новый диалог.')
        return json.loads(row['state'])

    def save(self, state, lead=None):
        # Lead and session are committed together: confirmation is idempotent.
        with self.connect() as db:
            if lead:
                db.execute('''INSERT OR IGNORE INTO leads
                    (id,session_id,full_name,phone,company,summary,created_at)
                    VALUES (:id,:session_id,:full_name,:phone,:company,:summary,:created_at)''', lead)
            db.execute('INSERT OR REPLACE INTO sessions VALUES (?,?,?)',
                       (state['id'], json.dumps(state, ensure_ascii=False), now()))

    def list_sessions(self, limit=50):
        """Return recent sessions for the explicitly enabled local debug view."""
        with self.connect() as db:
            rows = db.execute(
                'SELECT id,state,updated_at FROM sessions ORDER BY updated_at DESC LIMIT ?',
                (max(1, min(int(limit), 200)),),
            ).fetchall()
        result = []
        for row in rows:
            state = json.loads(row['state'])
            result.append({
                'session_ref': row['id'][:8],
                'updated_at': row['updated_at'],
                'has_lead': bool(state.get('lead_id')),
                'history': state.get('history', []),
            })
        return result
