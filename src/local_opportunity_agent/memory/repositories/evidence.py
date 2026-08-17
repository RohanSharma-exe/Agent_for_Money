from __future__ import annotations

import sqlite3


class EvidenceRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def create(
        self,
        *,
        evidence_id: str,
        source_id: str,
        evidence_type: str,
        content: str,
        created_at: str,
        opportunity_id: str | None = None,
        quote: str | None = None,
        confidence: float | None = None,
        metadata_json: str | None = None,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO evidence (
                id,
                source_id,
                opportunity_id,
                evidence_type,
                content,
                quote,
                confidence,
                metadata_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                evidence_id,
                source_id,
                opportunity_id,
                evidence_type,
                content,
                quote,
                confidence,
                metadata_json,
                created_at,
            ),
        )

    def for_opportunity(
        self,
        opportunity_id: str,
    ) -> list[sqlite3.Row]:
        return self.connection.execute(
            """
            SELECT *
            FROM evidence
            WHERE opportunity_id = ?
            ORDER BY created_at ASC
            """,
            (opportunity_id,),
        ).fetchall()

    def for_source(
        self,
        source_id: str,
    ) -> list[sqlite3.Row]:
        return self.connection.execute(
            """
            SELECT *
            FROM evidence
            WHERE source_id = ?
            ORDER BY created_at ASC
            """,
            (source_id,),
        ).fetchall()
