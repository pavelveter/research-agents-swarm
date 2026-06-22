"""Swarm Memory Bank — Qdrant backend for semantic deduplication with native Ollama client."""

from __future__ import annotations

import asyncio
import logging
import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from http_client import get_http_client

logger = logging.getLogger(__name__)


class SwarmMemoryBank:
    """Enterprise memory bank leveraging AsyncQdrantClient for native async/await graph execution."""

    def __init__(self, qdrant_url: str = "http://127.0.0.1:6333", ollama_url: str = "http://127.0.0.1:11434/api/embeddings") -> None:
        self.client = AsyncQdrantClient(url=qdrant_url, api_key=None)
        self.collection_name = "swarm_validated_facts"

        self.ollama_url = ollama_url
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
        client = get_http_client()
        response = await client.post(
            self.ollama_url, json={"model": self.embedding_model, "prompt": text}
        )
        if response.status_code != 200:
            logger.error("Ollama embedding error: %s", response.text)
            response.raise_for_status()

        data = response.json()
        return list(data["embedding"])

    _EMBED_SEMAPHORE = asyncio.Semaphore(8)

    async def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Batch embed documents concurrently via Ollama with bounded parallelism."""

        async def _guarded_embed(text: str) -> list[float]:
            async with self._EMBED_SEMAPHORE:
                return await self._embed_string(text)

        return list(await asyncio.gather(*[_guarded_embed(t) for t in texts]))

    async def upsert_facts(
        self,
        facts: list[str],
        iteration: int,
        task: str,
        threshold: float = 0.92,
        layer: str = "analysis",
    ) -> int:
        """Embed and store unique facts, executing strict semantic deduplication.

        T14: ``layer`` tags facts as "principles" (official/regulatory) or
        "analysis" (analytical/preprint). Defaults to "analysis".
        """
        if not facts:
            return 0

        await self._ensure_collection()

        vectors = await self._embed_documents(facts)
        points = []
        new_facts_count = 0

        for fact, vector in zip(facts, vectors):
            if await self._is_semantic_duplicate(vector, threshold=threshold):
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
                        "layer": layer,
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

    async def retrieve_context(
        self, query: str, limit: int = 20, layer: str | None = None
    ) -> list[str]:
        """Fetch closest context chunks based on semantic distance.

        T14: When ``layer`` is provided, results are split into two passes:
        Pass 1 — Layer "principles" (core texts), Pass 2 — Layer "analysis".
        Results are interleaved so principles appear first, then analysis.
        """
        await self._ensure_collection()
        vector = await self._embed_string(query)

        if layer:
            # Two-pass retrieval: principles first, then analysis
            principles_limit = max(1, limit // 2)
            analysis_limit = limit - principles_limit

            principles_hits = await self._query_with_filter(
                vector, principles_limit, layer_tag="principles"
            )
            analysis_hits = await self._query_with_filter(
                vector, analysis_limit, layer_tag="analysis"
            )
            # Interleave: principles first, then supplement with analysis
            all_hits = principles_hits + analysis_hits
            return [
                str(hit.payload["fact"])
                for hit in all_hits
                if hit.payload and "fact" in hit.payload
            ]

        # Single-pass retrieval (no layer preference)
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

    async def _query_with_filter(
        self,
        vector: list[float],
        limit: int,
        layer_tag: str,
    ) -> list:
        """Query Qdrant with a payload filter on the 'layer' field."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        response = await self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=limit,
            query_filter=Filter(
                must=[FieldCondition(key="layer", match=MatchValue(value=layer_tag))]
            ),
        )
        return list(response.points)

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
        from config.settings import get_settings
        s = get_settings()
        _memory_bank = SwarmMemoryBank(qdrant_url=s.qdrant_url, ollama_url=s.ollama_url)
    return _memory_bank
