

import chromadb
from pathlib import Path
from chromadb.utils import embedding_functions

DATA_DIR = Path("data")
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
CHROMA_DIR = DATA_DIR / "chroma_db"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    ef = embedding_functions.DefaultEmbeddingFunction()
    return client.get_or_create_collection(
        name="earnings_transcripts",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


def chunk_text(text: str) -> list:
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + CHUNK_SIZE]))
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def clean_transcript(raw: str) -> str:
    lines = [l.strip() for l in raw.splitlines() if len(l.strip()) > 20]
    return " ".join(lines)


def ingest_transcript(ticker: str, date: str, text: str, collection) -> int:
    chunks = chunk_text(clean_transcript(text))
    if not chunks:
        return 0
    ids = [f"{ticker}_{date}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"ticker": ticker, "date": date, "chunk_index": i} for i in range(len(chunks))]
    try:
        collection.delete(where={"ticker": ticker, "date": date})
    except Exception:
        pass
    collection.add(ids=ids, documents=chunks, metadatas=metadatas)
    return len(chunks)


def query_transcript(query: str, ticker: str = None, n_results: int = 5) -> list:
    collection = get_collection()
    where = {"ticker": ticker} if ticker else None
    results = collection.query(query_texts=[query], n_results=n_results, where=where)
    return [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


def embed_all_transcripts():
    collection = get_collection()
    files = list(TRANSCRIPTS_DIR.glob("*.txt"))

    if not files:
        print("No transcripts found. Run: python3 scraper.py --sample")
        return

    total = 0
    for path in files:
        parts = path.stem.split("_", 1)
        if len(parts) != 2:
            continue
        ticker, date = parts
        text = path.read_text(encoding="utf-8")
        print(f"  Ingesting {ticker} {date} ({len(text.split())} words)...", end=" ")
        n = ingest_transcript(ticker, date, text, collection)
        print(f"{n} chunks")
        total += n

    print(f"\nDone. {total} chunks stored in ChromaDB.")


if __name__ == "__main__":
    embed_all_transcripts()
    print("\nTest query: 'revenue guidance next quarter'")
    results = query_transcript("revenue guidance next quarter", n_results=3)
    for r in results:
        print(f"  [{r['metadata']['ticker']} {r['metadata']['date']}] {r['text'][:120]}...")
