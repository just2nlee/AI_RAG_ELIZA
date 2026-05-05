#!/usr/bin/env python3
"""
One-time indexing script: chunks all SEC filings in edgar_corpus/, embeds them
with OpenAI text-embedding-3-small, and persists the vectors to ChromaDB on disk.

Run this once before starting the backend. Re-run only if the corpus changes.
Usage: python index.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

sys.path.insert(0, "backend")
from retrieval import chunk_text, parse_filename

CORPUS_DIR = Path("edgar_corpus")
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "sec_filings"
# OpenAI's embedding API accepts up to 2048 inputs per request; 100 is a safe
# batch size that avoids hitting token-count limits on large chunks.
BATCH_SIZE = 100


def embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts and return their vectors."""
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [item.embedding for item in resp.data]


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit("OPENAI_API_KEY not set in environment or .env file")

    openai_client = OpenAI(api_key=api_key)

    # PersistentClient writes vectors + metadata to disk so they survive restarts.
    # The backend loads this same directory at startup — no re-embedding needed.
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        # Cosine similarity is standard for text embeddings; dot-product would
        # require normalized vectors and Euclidean doesn't suit high-dim text.
        metadata={"hnsw:space": "cosine"},
    )

    with open(CORPUS_DIR / "manifest.json") as f:
        manifest = json.load(f)

    filenames: list[str] = manifest["files"]
    print(f"Indexing {len(filenames)} filings...")

    for i, filename in enumerate(filenames):
        filepath = CORPUS_DIR / filename
        if not filepath.exists():
            print(f"  SKIP (missing): {filename}")
            continue

        meta = parse_filename(filename)
        text = filepath.read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_text(text)

        for batch_start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[batch_start : batch_start + BATCH_SIZE]
            embeddings = embed_batch(openai_client, batch)
            # IDs must be unique across the entire collection; using the filename
            # stem + position ensures no collisions across filings.
            ids = [f"{filepath.stem}_{batch_start + j}" for j in range(len(batch))]
            metadatas = [
                {
                    "ticker": meta["ticker"],
                    "filing_type": meta["filing_type"],
                    "period": meta["period"],
                    "filename": filename,
                }
                for _ in batch
            ]
            collection.add(
                embeddings=embeddings,
                documents=batch,
                metadatas=metadatas,
                ids=ids,
            )

        print(f"  [{i+1}/{len(filenames)}] {filename}: {len(chunks)} chunks")

    print(f"\nDone. Collection has {collection.count()} chunks total.")


if __name__ == "__main__":
    main()
