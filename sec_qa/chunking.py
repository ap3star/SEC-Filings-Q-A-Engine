from langchain_text_splitters import RecursiveCharacterTextSplitter

from .models import Chunk

_splitter = RecursiveCharacterTextSplitter(chunk_size=1800, chunk_overlap=200)


def chunk_sections(sections, ticker, doc_type, period, source) -> list[Chunk]:
    chunks = []
    for section, body in sections:
        for i, piece in enumerate(_splitter.split_text(body)):
            if len(piece.strip()) < 80:
                continue
            chunks.append(Chunk(
                chunk_id=f"{ticker}-{doc_type}-{period}-{section}-{i}".replace(" ", "_"),
                text=piece, ticker=ticker, doc_type=doc_type,
                period=period, section=section, source=source,
            ))
    return chunks
