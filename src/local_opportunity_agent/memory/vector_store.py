from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import (
    ApiException,
    ResponseHandlingException,
)
from qdrant_client.models import (
    Distance,
    PointIdsList,
    PointStruct,
    VectorParams,
)


class VectorStoreError(RuntimeError):
    """Raised when a vector-store operation fails."""


@dataclass(frozen=True)
class SearchResult:
    point_id: str
    score: float
    payload: dict


class VectorStore:
    """Persistent local Qdrant Edge vector store."""

    def __init__(
        self,
        *,
        path: Path,
        collection_name: str,
        vector_size: int,
    ) -> None:
        self.path = path
        self.collection_name = collection_name
        self.vector_size = vector_size

        self.path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._client = QdrantClient(
            path=str(self.path),
        )

    def initialize(self) -> None:
        """Create the collection if it does not already exist."""
        try:
            collections = self._client.get_collections()

            names = {collection.name for collection in collections.collections}

            if self.collection_name in names:
                return

            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )
        except Exception as error:
            raise VectorStoreError(f"Unable to initialize Qdrant collection: {error}") from error

    def upsert(
        self,
        *,
        point_id: str,
        vector: list[float],
        payload: dict,
    ) -> None:
        """Insert or replace a vector."""
        self._validate_vector(vector)

        try:
            self._client.upsert(
                collection_name=self.collection_name,
                wait=True,
                points=[
                    PointStruct(
                        id=str(
                            uuid.uuid5(
                                uuid.NAMESPACE_URL,
                                f"local-opportunity-agent:{point_id}",
                            )
                        ),
                        vector=vector,
                        payload={
                            **payload,
                            "_source_id": point_id,
                        },
                    )
                ],
            )
        except Exception as error:
            raise VectorStoreError(f"Unable to upsert vector '{point_id}': {error}") from error

    def delete(self, point_id: str) -> None:
        """Delete a vector by its application-level ID."""
        qdrant_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"local-opportunity-agent:{point_id}",
            )
        )

        try:
            self._client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(
                    points=[qdrant_id],
                ),
                wait=True,
            )
        except (
            ApiException,
            ResponseHandlingException,
        ) as error:
            raise VectorStoreError(f"Unable to delete vector '{point_id}': {error}") from error

    def search(
        self,
        *,
        vector: list[float],
        limit: int = 5,
    ) -> list[SearchResult]:
        """Perform semantic similarity search."""
        self._validate_vector(vector)

        if limit < 1:
            raise ValueError("limit must be at least 1")

        try:
            results = self._client.query_points(
                collection_name=self.collection_name,
                query=vector,
                limit=limit,
                with_payload=True,
            )
        except Exception as error:
            raise VectorStoreError(f"Unable to search Qdrant: {error}") from error

        return [
            SearchResult(
                point_id=str(
                    point.payload.get(
                        "_source_id",
                        point.id,
                    )
                ),
                score=float(point.score),
                payload=dict(point.payload or {}),
            )
            for point in results.points
        ]

    def count(self) -> int:
        """Return the number of stored vectors."""
        try:
            result = self._client.count(
                collection_name=self.collection_name,
                exact=True,
            )
        except Exception as error:
            raise VectorStoreError(f"Unable to count Qdrant vectors: {error}") from error

        return int(result.count)

    def health_check(self) -> bool:
        """Verify the collection can be accessed."""
        try:
            self._client.get_collection(
                self.collection_name,
            )
        except (
            ApiException,
            ResponseHandlingException,
        ):
            return False

        return True

    def _validate_vector(
        self,
        vector: list[float],
    ) -> None:
        if len(vector) != self.vector_size:
            raise ValueError(
                f"Expected vector with {self.vector_size} dimensions, received {len(vector)}"
            )

    def close(self) -> None:
        """Close the local Qdrant client."""
        self._client.close()
