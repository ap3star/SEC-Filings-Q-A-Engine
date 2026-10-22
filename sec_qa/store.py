"""Elasticsearch index with hybrid (BM25 + dense vector) retrieval."""
from elasticsearch import Elasticsearch, helpers

from .config import EMBED_DIM, ES_INDEX, ES_URL
from .embeddings import embed_query, embed_texts
from .models import Chunk

MAPPING = {
    "properties": {
        "text": {"type": "text"},
        "ticker": {"type": "keyword"},
        "doc_type": {"type": "keyword"},
        "period": {"type": "keyword"},
        "section": {"type": "keyword"},
        "source": {"type": "keyword"},
        "embedding": {"type": "dense_vector", "dims": EMBED_DIM,
                      "index": True, "similarity": "cosine"},
    }
}


def client() -> Elasticsearch:
    return Elasticsearch(ES_URL)


def create_index(es: Elasticsearch, recreate: bool = False) -> None:
    if recreate and es.indices.exists(index=ES_INDEX):
        es.indices.delete(index=ES_INDEX)
    if not es.indices.exists(index=ES_INDEX):
        es.indices.create(index=ES_INDEX, mappings=MAPPING)


def index_chunks(chunks: list[Chunk], batch: int = 64, recreate: bool = False) -> int:
    es = client()
    create_index(es, recreate)
    for i in range(0, len(chunks), batch):
        part = chunks[i:i + batch]
        vecs = embed_texts([c.text for c in part])
        actions = []
        for c, v in zip(part, vecs):
            doc = c.to_dict()
            doc["embedding"] = v
            actions.append({"_index": ES_INDEX, "_id": c.chunk_id, "_source": doc})
        helpers.bulk(es, actions)
    es.indices.refresh(index=ES_INDEX)
    return len(chunks)


def build_query(question: str, vector: list[float], k: int,
                tickers: list[str] | None = None, doc_type: str | None = None) -> dict:
    filters = []
    if tickers:
        filters.append({"terms": {"ticker": tickers}})
    if doc_type:
        filters.append({"term": {"doc_type": doc_type}})
    return {
        "size": k,
        "query": {"bool": {"must": [{"match": {"text": {"query": question, "boost": 0.4}}}],
                           "filter": filters}},
        "knn": {"field": "embedding", "query_vector": vector, "k": k,
                "num_candidates": max(100, k * 10), "filter": filters, "boost": 0.6},
        "_source": {"excludes": ["embedding"]},
    }


def search(question: str, k: int = 8, tickers: list[str] | None = None,
           doc_type: str | None = None) -> list[dict]:
    body = build_query(question, embed_query(question), k, tickers, doc_type)
    res = client().search(index=ES_INDEX, **body)
    return [{**h["_source"], "score": h["_score"]} for h in res["hits"]["hits"]]
