from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS media (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL UNIQUE,
    filename TEXT NOT NULL,
    extension TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    duration_seconds REAL,
    status TEXT NOT NULL DEFAULT 'pending',
    error_message TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS audio_chunk (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    media_id INTEGER NOT NULL REFERENCES media(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    start_seconds REAL NOT NULL,
    end_seconds REAL NOT NULL,
    audio_path TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    error_message TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(media_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS transcript_segment (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    media_id INTEGER NOT NULL REFERENCES media(id) ON DELETE CASCADE,
    chunk_id INTEGER REFERENCES audio_chunk(id) ON DELETE SET NULL,
    start_seconds REAL NOT NULL,
    end_seconds REAL NOT NULL,
    text TEXT NOT NULL,
    normalized_text TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS transcript_word (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    segment_id INTEGER NOT NULL REFERENCES transcript_segment(id) ON DELETE CASCADE,
    media_id INTEGER NOT NULL REFERENCES media(id) ON DELETE CASCADE,
    word TEXT NOT NULL,
    normalized_word TEXT NOT NULL,
    start_seconds REAL NOT NULL,
    end_seconds REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS transcript_embedding (
    segment_id INTEGER NOT NULL REFERENCES transcript_segment(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    vector_norm REAL NOT NULL,
    vector BLOB NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(segment_id, model)
);

CREATE TABLE IF NOT EXISTS app_event (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    level TEXT NOT NULL,
    message TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS processing_progress (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    active INTEGER NOT NULL DEFAULT 0,
    stage TEXT NOT NULL DEFAULT 'idle',
    media_id INTEGER,
    filename TEXT,
    current_chunk INTEGER NOT NULL DEFAULT 0,
    total_chunks INTEGER NOT NULL DEFAULT 0,
    current_seconds REAL NOT NULL DEFAULT 0,
    total_seconds REAL NOT NULL DEFAULT 0,
    percent REAL NOT NULL DEFAULT 0,
    message TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE VIRTUAL TABLE IF NOT EXISTS transcript_fts
USING fts5(text, normalized_text, content='transcript_segment', content_rowid='id');

CREATE TRIGGER IF NOT EXISTS transcript_segment_ai AFTER INSERT ON transcript_segment BEGIN
    INSERT INTO transcript_fts(rowid, text, normalized_text)
    VALUES (new.id, new.text, new.normalized_text);
END;

CREATE TRIGGER IF NOT EXISTS transcript_segment_ad AFTER DELETE ON transcript_segment BEGIN
    INSERT INTO transcript_fts(transcript_fts, rowid, text, normalized_text)
    VALUES ('delete', old.id, old.text, old.normalized_text);
END;

CREATE TRIGGER IF NOT EXISTS transcript_segment_au AFTER UPDATE ON transcript_segment BEGIN
    INSERT INTO transcript_fts(transcript_fts, rowid, text, normalized_text)
    VALUES ('delete', old.id, old.text, old.normalized_text);
    INSERT INTO transcript_fts(rowid, text, normalized_text)
    VALUES (new.id, new.text, new.normalized_text);
END;

CREATE INDEX IF NOT EXISTS idx_media_status ON media(status);
CREATE INDEX IF NOT EXISTS idx_chunk_media ON audio_chunk(media_id);
CREATE INDEX IF NOT EXISTS idx_chunk_status ON audio_chunk(status);
CREATE INDEX IF NOT EXISTS idx_segment_media_time ON transcript_segment(media_id, start_seconds);
CREATE INDEX IF NOT EXISTS idx_word_media_time ON transcript_word(media_id, start_seconds);
CREATE INDEX IF NOT EXISTS idx_embedding_model ON transcript_embedding(model);
"""


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.init()

    def init(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            self._ensure_fts(conn)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _ensure_fts(self, conn: sqlite3.Connection) -> None:
        count = conn.execute("SELECT COUNT(*) FROM transcript_fts").fetchone()[0]
        if count == 0:
            conn.execute(
                """
                INSERT INTO transcript_fts(rowid, text, normalized_text)
                SELECT id, text, normalized_text FROM transcript_segment
                """
            )


def row_to_dict(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    return {key: row[key] for key in row.keys()}
