from langchain_openai import OpenAIEmbeddings

from .config import EMBED_MODEL

_emb = None


def embedder() -> OpenAIEmbeddings:
    global _emb
    if _emb is None:
        _emb = OpenAIEmbeddings(model=EMBED_MODEL)
    return _emb


def embed_texts(texts: list[str]) -> list[list[float]]:
    return embedder().embed_documents(texts)


def embed_query(text: str) -> list[float]:
    return embedder().embed_query(text)
