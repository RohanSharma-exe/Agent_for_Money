from __future__ import annotations

import sqlite3


class OpportunityRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def create(
        self,
        *,
        opportunity_id: str,
        title: str,
        description: str | None,
        created_at: str,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO opportunities (
                id,
                title,
                description,
                status,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, 'candidate', ?, ?)
            """,
            (
                opportunity_id,
                title,
                description,
                created_at,
                created_at,
            ),
        )

    def get(self, opportunity_id: str) -> sqlite3.Row | None:
        return self.connection.execute(
            """
            SELECT *
            FROM opportunities
            WHERE id = ?
            """,
            (opportunity_id,),
        ).fetchone()

    def list_candidates(self) -> list[sqlite3.Row]:
        return self.connection.execute(
            """
            SELECT *
            FROM opportunities
            WHERE status = 'candidate'
            ORDER BY created_at DESC
            """
        ).fetchall()

    def update_score(
        self,
        *,
        opportunity_id: str,
        score: float,
        updated_at: str,
    ) -> None:
        self.connection.execute(
            """
            UPDATE opportunities
            SET score = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                score,
                updated_at,
                opportunity_id,
            ),
        )
