from __future__ import annotations

import sqlite3


class SourceRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def create(
        self,
        *,
        source_id: str,
        source_type: str,
        title: str | None,
        url: str | None,
        created_at: str,
        author: str | None = None,
        published_at: str | None = None,
        content_hash: str | None = None,
        metadata_json: str | None = None,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO sources (
                id,
                source_type,
                title,
                url,
                author,
                published_at,
                content_hash,
                metadata_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source_id,
                source_type,
                title,
                url,
                author,
                published_at,
                content_hash,
                metadata_json,
                created_at,
            ),
        )

    def get(self, source_id: str) -> sqlite3.Row | None:
        return self.connection.execute(
            """
            SELECT *
            FROM sources
            WHERE id = ?
            """,
            (source_id,),
        ).fetchone()
