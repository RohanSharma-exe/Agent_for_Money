from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from local_opportunity_agent.memory.database import Database


class DatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "test.db"

        self.database = Database(self.database_path)
        self.database.initialize()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_database_file_is_created(self) -> None:
        self.assertTrue(self.database_path.exists())

    def test_schema_version_is_one(self) -> None:
        self.assertEqual(
            self.database.get_schema_version(),
            1,
        )

    def test_health_check(self) -> None:
        self.assertTrue(self.database.health_check())

    def test_required_tables_exist(self) -> None:
        expected_tables = {
            "schema_metadata",
            "opportunities",
            "sources",
            "evidence",
            "research_runs",
            "agent_runs",
            "memory",
        }

        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                """
            ).fetchall()

        actual_tables = {row["name"] for row in rows}

        self.assertTrue(expected_tables.issubset(actual_tables))

    def test_foreign_keys_are_enabled(self) -> None:
        with self.database.connection() as connection:
            row = connection.execute("PRAGMA foreign_keys").fetchone()

        self.assertEqual(row[0], 1)

    def test_foreign_key_relationship_works(self) -> None:
        with self.database.connection() as connection:
            connection.execute(
                """
                INSERT INTO opportunities (
                    id,
                    title,
                    description,
                    status,
                    created_at,
                    updated_at
                )
                VALUES (
                    'opp-1',
                    'Test opportunity',
                    'Test description',
                    'candidate',
                    '2026-08-17T00:00:00Z',
                    '2026-08-17T00:00:00Z'
                )
                """
            )

            connection.execute(
                """
                INSERT INTO sources (
                    id,
                    source_type,
                    title,
                    created_at
                )
                VALUES (
                    'src-1',
                    'test',
                    'Test source',
                    '2026-08-17T00:00:00Z'
                )
                """
            )

            connection.execute(
                """
                INSERT INTO evidence (
                    id,
                    source_id,
                    opportunity_id,
                    evidence_type,
                    content,
                    created_at
                )
                VALUES (
                    'ev-1',
                    'src-1',
                    'opp-1',
                    'observation',
                    'Test evidence',
                    '2026-08-17T00:00:00Z'
                )
                """
            )

            row = connection.execute(
                """
                SELECT evidence.id
                FROM evidence
                JOIN sources
                    ON evidence.source_id = sources.id
                JOIN opportunities
                    ON evidence.opportunity_id = opportunities.id
                WHERE evidence.id = 'ev-1'
                """
            ).fetchone()

        self.assertIsNotNone(row)

    def test_invalid_foreign_key_is_rejected(self) -> None:
        with (
            self.assertRaises(sqlite3.IntegrityError),
            self.database.connection() as connection,
        ):
            connection.execute(
                """
                    INSERT INTO evidence (
                        id,
                        source_id,
                        evidence_type,
                        content,
                        created_at
                    )
                    VALUES (
                        'ev-invalid',
                        'missing-source',
                        'observation',
                        'Invalid evidence',
                        '2026-08-17T00:00:00Z'
                    )
                    """
            )


if __name__ == "__main__":
    unittest.main()
