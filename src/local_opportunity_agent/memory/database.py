from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SCHEMA_VERSION = 1


class Database:
    """Small SQLite database wrapper for structured application state."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def initialize(self) -> None:
        """Create the database and all required tables."""
        self.path.parent.mkdir(parents=True, exist_ok=True)

        with self.connection() as connection:
            connection.executescript(
                """
                PRAGMA foreign_keys = ON;

                CREATE TABLE IF NOT EXISTS schema_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                INSERT OR REPLACE INTO schema_metadata (key, value)
                VALUES ('schema_version', '1');

                CREATE TABLE IF NOT EXISTS opportunities (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    status TEXT NOT NULL DEFAULT 'candidate',
                    score REAL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sources (
                    id TEXT PRIMARY KEY,
                    source_type TEXT NOT NULL,
                    title TEXT,
                    url TEXT,
                    author TEXT,
                    published_at TEXT,
                    content_hash TEXT,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS evidence (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    opportunity_id TEXT,
                    evidence_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    quote TEXT,
                    confidence REAL,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL,

                    FOREIGN KEY (source_id)
                        REFERENCES sources(id)
                        ON DELETE CASCADE,

                    FOREIGN KEY (opportunity_id)
                        REFERENCES opportunities(id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS research_runs (
                    id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    metadata_json TEXT
                );

                CREATE TABLE IF NOT EXISTS agent_runs (
                    id TEXT PRIMARY KEY,
                    agent_name TEXT NOT NULL,
                    research_run_id TEXT,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    input_json TEXT,
                    output_json TEXT,
                    error TEXT,

                    FOREIGN KEY (research_run_id)
                        REFERENCES research_runs(id)
                        ON DELETE SET NULL
                );

                CREATE TABLE IF NOT EXISTS memory (
                    id TEXT PRIMARY KEY,
                    memory_type TEXT NOT NULL,
                    subject TEXT,
                    content TEXT NOT NULL,
                    source_id TEXT,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,

                    FOREIGN KEY (source_id)
                        REFERENCES sources(id)
                        ON DELETE SET NULL
                );

                CREATE INDEX IF NOT EXISTS idx_opportunities_status
                    ON opportunities(status);

                CREATE INDEX IF NOT EXISTS idx_sources_type
                    ON sources(source_type);

                CREATE INDEX IF NOT EXISTS idx_evidence_source
                    ON evidence(source_id);

                CREATE INDEX IF NOT EXISTS idx_evidence_opportunity
                    ON evidence(opportunity_id);

                CREATE INDEX IF NOT EXISTS idx_agent_runs_research
                    ON agent_runs(research_run_id);

                CREATE INDEX IF NOT EXISTS idx_memory_type
                    ON memory(memory_type);
                """
            )

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Yield a configured SQLite connection."""
        connection = sqlite3.connect(self.path)

        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 5000")

        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def health_check(self) -> bool:
        """Verify that SQLite can execute a simple query."""
        with self.connection() as connection:
            result = connection.execute("SELECT 1").fetchone()

        return result is not None and result[0] == 1

    def get_schema_version(self) -> int:
        """Return the current database schema version."""
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT value
                FROM schema_metadata
                WHERE key = 'schema_version'
                """
            ).fetchone()

        if row is None:
            return 0

        return int(row["value"])
