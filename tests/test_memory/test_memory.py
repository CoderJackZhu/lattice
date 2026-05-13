from __future__ import annotations

import asyncio
import json

import pytest

from lattice.memory.base import MemoryItem
from lattice.memory.working import WorkingMemory
from lattice.memory.episodic import EpisodicMemory
from lattice.memory.semantic import SemanticMemory
from lattice.memory.composite import CompositeMemory
from lattice.memory.stores.in_memory import InMemoryVectorStore, SearchResult, _cosine_similarity


# --- WorkingMemory ---

async def test_working_memory_store_retrieve():
    mem = WorkingMemory(max_items=10)
    await mem.store(MemoryItem(content="Python is great"))
    await mem.store(MemoryItem(content="Rust is fast"))

    results = await mem.retrieve("Python", top_k=5)
    assert len(results) == 1
    assert "Python" in results[0].content


async def test_working_memory_fifo():
    mem = WorkingMemory(max_items=3)
    for i in range(5):
        await mem.store(MemoryItem(content=f"item {i}"))

    results = await mem.retrieve("", top_k=10)
    assert len(results) == 3
    assert results[0].content == "item 2"
    assert results[2].content == "item 4"


async def test_working_memory_empty_query():
    mem = WorkingMemory()
    await mem.store(MemoryItem(content="a"))
    await mem.store(MemoryItem(content="b"))

    results = await mem.retrieve("", top_k=1)
    assert len(results) == 1
    assert results[0].content == "b"


async def test_working_memory_clear():
    mem = WorkingMemory()
    await mem.store(MemoryItem(content="data"))
    await mem.clear()
    results = await mem.retrieve("data")
    assert len(results) == 0


async def test_working_memory_source_set():
    mem = WorkingMemory()
    item = MemoryItem(content="test")
    await mem.store(item)
    assert item.source == "working"


# --- InMemoryVectorStore ---

async def _mock_embedding(texts: list[str]) -> list[list[float]]:
    return [[float(len(t)), float(len(t) % 3), 1.0] for t in texts]


def test_cosine_similarity():
    assert _cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)
    assert _cosine_similarity([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)
    assert _cosine_similarity([0, 0, 0], [1, 0, 0]) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        _cosine_similarity([1, 0], [1])


async def test_vector_store_add_search():
    store = InMemoryVectorStore(embedding_fn=_mock_embedding)
    ids = await store.add(["hello world", "goodbye"], [{"k": "v1"}, {"k": "v2"}])
    assert len(ids) == 2

    results = await store.search("hello world", top_k=2)
    assert len(results) == 2
    assert results[0].text in ("hello world", "goodbye")


async def test_vector_store_delete():
    store = InMemoryVectorStore(embedding_fn=_mock_embedding)
    ids = await store.add(["doc1", "doc2"], [{}, {}])
    await store.delete([ids[0]])

    results = await store.search("doc1", top_k=10)
    assert len(results) == 1


async def test_vector_store_empty_search():
    store = InMemoryVectorStore(embedding_fn=_mock_embedding)
    results = await store.search("query")
    assert results == []


async def test_vector_store_rejects_mismatched_input_lengths():
    store = InMemoryVectorStore(embedding_fn=_mock_embedding)
    with pytest.raises(ValueError):
        await store.add(["doc1", "doc2"], [{}])


async def test_vector_store_rejects_embedding_count_mismatch():
    async def bad_embedding(texts: list[str]) -> list[list[float]]:
        return [[1.0]]

    store = InMemoryVectorStore(embedding_fn=bad_embedding)
    with pytest.raises(ValueError):
        await store.add(["doc1", "doc2"], [{}, {}])


# --- EpisodicMemory ---

async def test_episodic_memory_store_retrieve(tmp_path):
    path = str(tmp_path / "ep.json")
    mem = EpisodicMemory(store_path=path)
    await mem.store(MemoryItem(content="solved bug in auth module"))
    await mem.store(MemoryItem(content="deployed to production"))

    results = await mem.retrieve("bug", top_k=5)
    assert len(results) == 1
    assert "bug" in results[0].content


async def test_episodic_memory_persistence(tmp_path):
    path = str(tmp_path / "ep.json")
    mem1 = EpisodicMemory(store_path=path)
    await mem1.store(MemoryItem(content="persistent data"))

    mem2 = EpisodicMemory(store_path=path)
    results = await mem2.retrieve("persistent", top_k=5)
    assert len(results) == 1


async def test_episodic_memory_with_embeddings(tmp_path):
    path = str(tmp_path / "ep_embed.json")
    mem = EpisodicMemory(store_path=path, embedding_fn=_mock_embedding)
    await mem.store(MemoryItem(content="vector search test"))

    results = await mem.retrieve("vector", top_k=5)
    assert len(results) >= 1


async def test_episodic_memory_clear(tmp_path):
    path = str(tmp_path / "ep_clear.json")
    mem = EpisodicMemory(store_path=path)
    await mem.store(MemoryItem(content="temp"))
    await mem.clear()
    results = await mem.retrieve("temp")
    assert len(results) == 0


# --- SemanticMemory ---

async def test_semantic_memory():
    store = InMemoryVectorStore(embedding_fn=_mock_embedding)
    mem = SemanticMemory(backend=store)

    await mem.store(MemoryItem(content="API documentation for auth"))
    results = await mem.retrieve("auth docs", top_k=5)
    assert len(results) == 1
    assert results[0].source == "semantic"


async def test_semantic_memory_clear_removes_stored_items():
    store = InMemoryVectorStore(embedding_fn=_mock_embedding)
    mem = SemanticMemory(backend=store)

    await mem.store(MemoryItem(content="API documentation for auth"))
    await mem.clear()

    assert await mem.retrieve("auth docs", top_k=5) == []


# --- CompositeMemory ---

async def test_composite_memory_retrieve():
    working = WorkingMemory()
    await working.store(MemoryItem(content="working item about Python"))

    ep_path = "/tmp/test_composite_ep.json"
    episodic = EpisodicMemory(store_path=ep_path)
    await episodic.store(MemoryItem(content="episodic item about Python"))

    composite = CompositeMemory([(working, 1.0), (episodic, 0.5)])
    results = await composite.retrieve("Python", top_k=5)

    assert len(results) == 2
    assert results[0].score >= results[1].score

    await episodic.clear()


async def test_composite_memory_store_primary():
    working = WorkingMemory()
    episodic_store = WorkingMemory()
    composite = CompositeMemory([(working, 1.0), (episodic_store, 0.5)])

    await composite.store(MemoryItem(content="stored to primary"))

    w_results = await working.retrieve("stored")
    e_results = await episodic_store.retrieve("stored")
    assert len(w_results) == 1
    assert len(e_results) == 0


async def test_composite_memory_clear():
    w = WorkingMemory()
    e = WorkingMemory()
    await w.store(MemoryItem(content="a"))
    await e.store(MemoryItem(content="b"))

    composite = CompositeMemory([(w, 1.0), (e, 0.5)])
    await composite.clear()

    assert len(await w.retrieve("")) == 0
    assert len(await e.retrieve("")) == 0
