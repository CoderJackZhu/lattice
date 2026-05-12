from __future__ import annotations

import asyncio
import uuid
from typing import Any

from lattice.memory.stores.in_memory import SearchResult, VectorStore


class ChromaVectorStore:
    def __init__(
        self, collection_name: str = "lattice", persist_directory: str | None = None
    ) -> None:
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "chromadb is required. Install with: pip install 'lattice[chroma]'"
            )

        if persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.EphemeralClient()
        self._collection = self._client.get_or_create_collection(collection_name)

    async def add(self, texts: list[str], metadatas: list[dict[str, Any]]) -> list[str]:
        ids = [uuid.uuid4().hex for _ in texts]
        await asyncio.to_thread(
            self._collection.add,
            documents=texts,
            metadatas=metadatas,
            ids=ids,
        )
        return ids

    async def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        results = await asyncio.to_thread(
            self._collection.query,
            query_texts=[query],
            n_results=top_k,
        )
        items: list[SearchResult] = []
        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas_list = results.get("metadatas", [[]])[0]

        for i, doc in enumerate(documents):
            score = 1.0 - (distances[i] if i < len(distances) else 0)
            metadata = metadatas_list[i] if i < len(metadatas_list) else {}
            items.append(SearchResult(text=doc, score=score, metadata=metadata))
        return items

    async def delete(self, ids: list[str]) -> None:
        await asyncio.to_thread(self._collection.delete, ids=ids)
