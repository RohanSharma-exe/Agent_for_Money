from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from local_opportunity_agent.memory import ObsidianMemory


class ObsidianMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.vault = Path(self.temp_dir.name)

        self.memory = ObsidianMemory(self.vault)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_initialize_creates_expected_directories(self) -> None:
        self.memory.initialize()

        for directory in (
            "opportunities",
            "research",
            "companies",
            "people",
        ):
            self.assertTrue((self.vault / directory).is_dir())

    def test_write_and_read_note(self) -> None:
        path = self.memory.write_note(
            category="opportunities",
            note_id="opp-001",
            title="AI Lead Automation",
            content="Automate lead qualification.",
            frontmatter={
                "id": "opp-001",
                "status": "candidate",
                "score": 8.5,
            },
        )

        self.assertTrue(path.exists())

        content = self.memory.read_note(
            category="opportunities",
            note_id="opp-001",
        )

        self.assertIsNotNone(content)

        assert content is not None

        self.assertIn(
            "# AI Lead Automation",
            content,
        )
        self.assertIn(
            "id: opp-001",
            content,
        )
        self.assertIn(
            "status: candidate",
            content,
        )
        self.assertIn(
            "score: 8.5",
            content,
        )
        self.assertIn(
            "Automate lead qualification.",
            content,
        )

    def test_write_note_sanitizes_path_components(self) -> None:
        path = self.memory.write_note(
            category="opportunities",
            note_id="test/unsafe note",
            title="Safe Note",
            content="Test",
        )

        self.assertTrue(path.exists())
        self.assertNotIn(
            "/",
            path.name,
        )

    def test_read_missing_note_returns_none(self) -> None:
        result = self.memory.read_note(
            category="opportunities",
            note_id="does-not-exist",
        )

        self.assertIsNone(result)

    def test_delete_note(self) -> None:
        self.memory.write_note(
            category="research",
            note_id="research-001",
            title="Research",
            content="Research content.",
        )

        deleted = self.memory.delete_note(
            category="research",
            note_id="research-001",
        )

        self.assertTrue(deleted)

        self.assertIsNone(
            self.memory.read_note(
                category="research",
                note_id="research-001",
            )
        )

    def test_delete_missing_note_returns_false(self) -> None:
        deleted = self.memory.delete_note(
            category="research",
            note_id="missing",
        )

        self.assertFalse(deleted)


if __name__ == "__main__":
    unittest.main()
