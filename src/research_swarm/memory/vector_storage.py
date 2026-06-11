"""Swarm Memory Bank — Qdrant backend for semantic deduplication with native Ollama client."""

from __future__ import annotations

import logging
import uuid

import httpx
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

logger = logging.getLogger(__name__)


class SwarmMemoryBank:
    """Enterprise memory bank leveraging AsyncQdrantClient for native async/await graph execution."""

    def __init__(self) -> None:
        qdrant_url = "http://127.0.0.1:6333"
        self.client = AsyncQdrantClient(url=qdrant_url, api_key=None)
        self.collection_name = "swarm_validated_facts"

        # Local Ollama configuration
        self.ollama_url = "http://127.0.0.1:11434/api/embeddings"
        self.embedding_model = "nomic-embed-text"

        self._collection_ready = False

    async def _ensure_collection(self, vector_size: int = 768) -> None:
        """Create Qdrant collection if missing using async boundaries."""
        if self._collection_ready:
            return
        try:
            exists = await self.client.collection_exists(self.collection_name)
            if not exists:
                await self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=vector_size, distance=Distance.COSINE
                    ),
                )
                logger.info(
                    "Successfully initialized Qdrant collection: %s with size %d",
                    self.collection_name,
                    vector_size,
                )
            self._collection_ready = True
        except Exception as exc:
            logger.error("Failed to check/create Qdrant collection: %s", exc)

    async def _embed_string(self, text: str) -> list[float]:
        """Generate embedding using native Ollama API."""
        async with httpx.AsyncClient(timeout=60.0) as http_client:
            response = await http_client.post(
                self.ollama_url, json={"model": self.embedding_model, "prompt": text}
            )
            if response.status_code != 200:
                logger.error("Ollama embedding error: %s", response.text)
                response.raise_for_status()

            data = response.json()
            return list(data["embedding"])

    async def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Batch embed documents via native sequential local execution."""
        return [await self._embed_string(text) for text in texts]

    async def upsert_facts(self, facts: list[str], iteration: int, task: str) -> int:
        """Embed and store unique facts, executing strict semantic deduplication."""
        if not facts:
            return 0

        await self._ensure_collection()

        vectors = await self._embed_documents(facts)
        points = []
        new_facts_count = 0

        for fact, vector in zip(facts, vectors):
            if await self._is_semantic_duplicate(vector, threshold=0.92):
                logger.debug("Skipping semantic duplicate fact: %s", fact[:60])
                continue

            point_id = str(uuid.uuid4())
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "fact": fact,
                        "iteration": iteration,
                        "task": task,
                    },
                )
            )
            new_facts_count += 1

        if points:
            # The upsert method in AsyncQdrantClient is asynchronous and has the same name
            await self.client.upsert(
                collection_name=self.collection_name, points=points
            )
            logger.info("Upserted %d new unique facts to Qdrant storage", len(points))

        return new_facts_count

    async def retrieve_context(self, query: str, limit: int = 20) -> list[str]:
        """Fetch closest context chunks based on semantic distance."""
        await self._ensure_collection()
        vector = await self._embed_string(query)

        # FIX: Replaced .search() with async method .query_points()
        response = await self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=limit,
        )
        return [
            str(hit.payload["fact"])
            for hit in response.points
            if hit.payload and "fact" in hit.payload
        ]

    async def _is_semantic_duplicate(
        self, text_or_vector: str | list[float], threshold: float
    ) -> bool:
        """Internal similarity scan. Accepts either raw string text or a pre-computed vector."""
        await self._ensure_collection()
        if isinstance(text_or_vector, str):
            vector = await self._embed_string(text_or_vector)
        else:
            vector = text_or_vector

        # FIX: Replaced .search() with async method .query_points()
        response = await self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=1,
        )
        if response.points and response.points[0].score >= threshold:
            return True
        return False


_memory_bank: SwarmMemoryBank | None = None


def get_memory_bank() -> SwarmMemoryBank:
    """Retrieve application-level global memory singleton."""
    global _memory_bank
    if _memory_bank is None:
        _memory_bank = SwarmMemoryBank()
    return _memory_bank
