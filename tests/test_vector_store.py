from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from local_opportunity_agent.memory.vector_store import (
    VectorStore,
    VectorStoreError,
)


class VectorStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()

        self.store = VectorStore(
            path=Path(self.temp_dir.name),
            collection_name="test_memory",
            vector_size=3,
        )

        self.store.initialize()

    def tearDown(self) -> None:
        self.store.close()
        self.temp_dir.cleanup()

    def test_initialize_creates_collection(self) -> None:
        self.assertTrue(self.store.health_check())

        self.assertEqual(
            self.store.count(),
            0,
        )

    def test_upsert_and_count(self) -> None:
        self.store.upsert(
            point_id="memory-1",
            vector=[1.0, 0.0, 0.0],
            payload={
                "type": "test",
                "text": "AI opportunity",
            },
        )

        self.assertEqual(
            self.store.count(),
            1,
        )

    def test_search_returns_original_application_id(self) -> None:
        self.store.upsert(
            point_id="memory-1",
            vector=[1.0, 0.0, 0.0],
            payload={
                "type": "test",
                "text": "AI opportunity",
            },
        )

        results = self.store.search(
            vector=[1.0, 0.0, 0.0],
            limit=1,
        )

        self.assertEqual(
            len(results),
            1,
        )

        self.assertEqual(
            results[0].point_id,
            "memory-1",
        )

        self.assertEqual(
            results[0].payload["text"],
            "AI opportunity",
        )

        self.assertGreater(
            results[0].score,
            0.99,
        )

    def test_upsert_replaces_existing_application_id(self) -> None:
        self.store.upsert(
            point_id="memory-1",
            vector=[1.0, 0.0, 0.0],
            payload={
                "version": 1,
            },
        )

        self.store.upsert(
            point_id="memory-1",
            vector=[0.0, 1.0, 0.0],
            payload={
                "version": 2,
            },
        )

        self.assertEqual(
            self.store.count(),
            1,
        )

        results = self.store.search(
            vector=[0.0, 1.0, 0.0],
            limit=1,
        )

        self.assertEqual(
            results[0].point_id,
            "memory-1",
        )

        self.assertEqual(
            results[0].payload["version"],
            2,
        )

    def test_delete_removes_vector(self) -> None:
        self.store.upsert(
            point_id="memory-1",
            vector=[1.0, 0.0, 0.0],
            payload={
                "type": "test",
            },
        )

        self.assertEqual(
            self.store.count(),
            1,
        )

        self.store.delete("memory-1")

        self.assertEqual(
            self.store.count(),
            0,
        )

    def test_wrong_vector_dimension_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.store.upsert(
                point_id="memory-1",
                vector=[1.0, 0.0],
                payload={},
            )

    def test_wrong_search_dimension_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.store.search(
                vector=[1.0, 0.0],
                limit=1,
            )

    def test_invalid_search_limit_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.store.search(
                vector=[1.0, 0.0, 0.0],
                limit=0,
            )

    def test_invalid_collection_operation_is_wrapped(self) -> None:
        broken_store = VectorStore(
            path=Path(self.temp_dir.name) / "broken",
            collection_name="missing_collection",
            vector_size=3,
        )

        with self.assertRaises(VectorStoreError):
            broken_store.search(
                vector=[1.0, 0.0, 0.0],
                limit=1,
            )

        broken_store.close()


if __name__ == "__main__":
    unittest.main()
