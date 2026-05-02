from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path

import tiktoken

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "text-embedding-3-small"
TOP_K = 15

_enc = tiktoken.get_encoding("cl100k_base")

KNOWN_TICKERS = {
    "AAPL", "ABBV", "ADBE", "AMD", "AMZN", "AXP", "BAC", "BA", "BLK",
    "BRK", "CAT", "CMCSA", "COST", "CRM", "CSCO", "CVX", "DE", "DIS",
    "GE", "GOOG", "GS", "HD", "IBM", "INTC", "JNJ", "JPM", "KO", "LLY",
    "LMT", "MA", "MCD", "META", "MRK", "MSFT", "NFLX", "NKE", "NVDA",
    "ORCL", "PEP", "PFE", "PG", "QCOM", "RTX", "SBUX", "T", "TSLA",
    "UNH", "UPS", "V", "VZ", "WMT", "XOM",
}


@dataclass
class Chunk:
    text: str
    ticker: str
    filing_type: str
    period: str


def parse_filename(filename: str) -> dict:
    """Parse ticker, filing_type, period from filenames like AAPL_10K_2022Q3_2022-10-28_full.txt"""
    stem = Path(filename).stem.replace("_full", "")
    parts = stem.split("_")

    ticker = parts[0]
    raw_type = parts[1]
    filing_type = "10-K" if raw_type == "10K" else "10-Q"

    # parts[2] is either a quarter period (2022Q3) or a date (2025-02-27)
    if len(parts) >= 3 and re.match(r"^\d{4}Q\d$", parts[2]):
        period = parts[2]
    else:
        period = parts[2][:4] if len(parts) >= 3 else "unknown"

    return {"ticker": ticker, "filing_type": filing_type, "period": period}


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping token-based chunks."""
    tokens = _enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(_enc.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start += chunk_size - overlap
    return chunks


def format_chunks(chunks: list[Chunk]) -> str:
    """Format chunks for injection into the LLM prompt."""
    parts = []
    for c in chunks:
        parts.append(f"[{c.ticker} {c.filing_type} {c.period}]\n{c.text}")
    return "\n\n---\n\n".join(parts)


def detect_tickers(query: str) -> list[str]:
    """Find known ticker symbols mentioned in the query."""
    query_upper = query.upper()
    found = []
    for ticker in KNOWN_TICKERS:
        if re.search(rf"\b{re.escape(ticker)}\b", query_upper):
            found.append(ticker)
    return found


def retrieve(query: str, openai_client, collection) -> list[Chunk]:
    """Retrieve top-k relevant chunks from ChromaDB."""
    tickers = detect_tickers(query)

    def embed(text: str) -> list[float]:
        resp = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=[text])
        return resp.data[0].embedding

    if len(tickers) > 1:
        per_ticker_k = max(5, TOP_K // len(tickers))
        seen: dict[str, Chunk] = {}
        for ticker in tickers:
            results = collection.query(
                query_embeddings=[embed(query)],
                n_results=per_ticker_k,
                where={"ticker": ticker},
            )
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                key = doc[:80]
                if key not in seen:
                    seen[key] = Chunk(
                        text=doc,
                        ticker=meta["ticker"],
                        filing_type=meta["filing_type"],
                        period=meta["period"],
                    )
        return list(seen.values())[:TOP_K]

    embedding = embed(query)
    kwargs: dict = {"query_embeddings": [embedding], "n_results": TOP_K}
    if len(tickers) == 1:
        kwargs["where"] = {"ticker": tickers[0]}
    results = collection.query(**kwargs)
    return [
        Chunk(
            text=doc,
            ticker=meta["ticker"],
            filing_type=meta["filing_type"],
            period=meta["period"],
        )
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]
