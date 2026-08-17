from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from local_opportunity_agent.memory.database import Database
from local_opportunity_agent.memory.repositories.evidence import (
    EvidenceRepository,
)
from local_opportunity_agent.memory.repositories.opportunities import (
    OpportunityRepository,
)
from local_opportunity_agent.memory.repositories.sources import (
    SourceRepository,
)


class RepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()

        database = Database(Path(self.temp_dir.name) / "test.db")
        database.initialize()

        self.connection_context = database.connection()
        self.connection = self.connection_context.__enter__()

        self.opportunities = OpportunityRepository(self.connection)
        self.sources = SourceRepository(self.connection)
        self.evidence = EvidenceRepository(self.connection)

    def tearDown(self) -> None:
        self.connection_context.__exit__(
            None,
            None,
            None,
        )
        self.temp_dir.cleanup()

    def test_opportunity_create_and_get(self) -> None:
        self.opportunities.create(
            opportunity_id="opp-1",
            title="AI lead follow-up",
            description="Automate lead follow-up.",
            created_at="2026-08-17T10:00:00Z",
        )

        opportunity = self.opportunities.get("opp-1")

        self.assertIsNotNone(opportunity)
        self.assertEqual(
            opportunity["title"],
            "AI lead follow-up",
        )

    def test_source_and_evidence_relationship(self) -> None:
        self.sources.create(
            source_id="src-1",
            source_type="web",
            title="Example source",
            url="https://example.com",
            created_at="2026-08-17T10:00:00Z",
        )

        self.opportunities.create(
            opportunity_id="opp-1",
            title="AI opportunity",
            description=None,
            created_at="2026-08-17T10:00:00Z",
        )

        self.evidence.create(
            evidence_id="ev-1",
            source_id="src-1",
            opportunity_id="opp-1",
            evidence_type="observation",
            content="Businesses complain about manual follow-up.",
            quote="Manual follow-up takes too much time.",
            confidence=0.9,
            created_at="2026-08-17T10:00:00Z",
        )

        evidence = self.evidence.for_opportunity("opp-1")

        self.assertEqual(len(evidence), 1)
        self.assertEqual(
            evidence[0]["source_id"],
            "src-1",
        )

    def test_candidate_listing(self) -> None:
        self.opportunities.create(
            opportunity_id="opp-1",
            title="Opportunity one",
            description=None,
            created_at="2026-08-17T10:00:00Z",
        )

        self.opportunities.create(
            opportunity_id="opp-2",
            title="Opportunity two",
            description=None,
            created_at="2026-08-17T11:00:00Z",
        )

        candidates = self.opportunities.list_candidates()

        self.assertEqual(len(candidates), 2)


if __name__ == "__main__":
    unittest.main()
