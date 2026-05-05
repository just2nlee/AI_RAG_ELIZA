from __future__ import annotations
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import tiktoken

CHUNK_SIZE = 500       # tokens per chunk
CHUNK_OVERLAP = 50     # tokens shared between adjacent chunks to preserve context across boundaries
EMBEDDING_MODEL = "text-embedding-3-small"
TOP_K = 15             # final number of chunks passed to the LLM

# cl100k_base is the tokenizer used by GPT-4 and text-embedding-3-small, so
# chunk boundaries here align exactly with what the model sees.
_enc = tiktoken.get_encoding("cl100k_base")

KNOWN_TICKERS = {
    "AAPL", "ABBV", "ADBE", "AMD", "AMZN", "AXP", "BAC", "BA", "BLK",
    "BRK", "CAT", "CMCSA", "COST", "CRM", "CSCO", "CVX", "DE", "DIS",
    "GE", "GOOG", "GS", "HD", "IBM", "INTC", "JNJ", "JPM", "KO", "LLY",
    "LMT", "MA", "MCD", "META", "MRK", "MSFT", "NFLX", "NKE", "NVDA",
    "ORCL", "PEP", "PFE", "PG", "QCOM", "RTX", "SBUX", "T", "TSLA",
    "UNH", "UPS", "V", "VZ", "WMT", "XOM",
}

# Plain-English name → ticker mapping. Users write "Apple" or "JPMorgan", not
# "AAPL" or "JPM". Without this, detect_tickers returns nothing for natural
# language queries and the per-ticker balanced retrieval logic never fires.
NAME_TO_TICKER: dict[str, str] = {
    "APPLE": "AAPL", "MICROSOFT": "MSFT", "AMAZON": "AMZN", "NVIDIA": "NVDA",
    "GOOGLE": "GOOG", "ALPHABET": "GOOG", "META": "META", "TESLA": "TSLA",
    "JPMORGAN": "JPM", "JP MORGAN": "JPM", "CHASE": "JPM",
    "BERKSHIRE": "BRK", "VISA": "V", "MASTERCARD": "MA",
    "WALMART": "WMT", "EXXON": "XOM", "JOHNSON": "JNJ",
    "UNITEDHEALTH": "UNH", "ABBVIE": "ABBV", "PFIZER": "PFE",
    "MERCK": "MRK", "ELI LILLY": "LLY", "LILLY": "LLY",
    "BROADCOM": "AVGO", "NETFLIX": "NFLX", "ADOBE": "ADBE",
    "SALESFORCE": "CRM", "ORACLE": "ORCL", "INTEL": "INTC",
    "QUALCOMM": "QCOM", "AMD": "AMD", "GOLDMAN": "GS",
    "BANK OF AMERICA": "BAC", "BLACKROCK": "BLK", "CATERPILLAR": "CAT",
    "BOEING": "BA", "COMCAST": "CMCSA", "COSTCO": "COST",
    "HOME DEPOT": "HD", "MCDONALD": "MCD", "NIKE": "NKE",
    "PEPSI": "PEP", "PEPSICO": "PEP", "PROCTER": "PG",
    "RAYTHEON": "RTX", "STARBUCKS": "SBUX", "AT&T": "T",
    "UPS": "UPS", "VERIZON": "VZ", "DISNEY": "DIS",
    "GENERAL ELECTRIC": "GE", "DEERE": "DE", "IBMS": "IBM",
    "CHEVRON": "CVX", "CISCO": "CSCO", "AMERICAN EXPRESS": "AXP",
    "LOCKHEED": "LMT",
}


@dataclass
class Chunk:
    text: str
    ticker: str
    filing_type: str
    period: str


def parse_filename(filename: str) -> dict:
    """Parse ticker, filing_type, and period from corpus filenames.

    Two filename formats exist in the corpus:
      - AAPL_10K_2022Q3_2022-10-28_full.txt  (period explicitly present)
      - GS_10K_2025-02-27_full.txt           (no period; year extracted from date)
    """
    stem = Path(filename).stem.replace("_full", "")
    parts = stem.split("_")

    ticker = parts[0]
    raw_type = parts[1]
    filing_type = "10-K" if raw_type == "10K" else "10-Q"

    # parts[2] is either a quarter period (e.g. "2022Q3") or a date (e.g. "2025-02-27")
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
    """Split text into overlapping token-based chunks.

    Token-based splitting (vs. character or sentence) ensures each chunk fits
    within the embedding model's context window regardless of word length.
    Overlap preserves context for sentences that straddle chunk boundaries.
    """
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
    """Serialize chunks into the labeled block format injected into the LLM prompt.

    Each block is prefixed with [TICKER FILING_TYPE PERIOD] so the model can
    produce inline citations that map back to specific filings.
    """
    parts = []
    for c in chunks:
        parts.append(f"[{c.ticker} {c.filing_type} {c.period}]\n{c.text}")
    return "\n\n---\n\n".join(parts)


_YEAR_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}


def parse_time_window(query: str) -> int | None:
    """Return the inclusive cutoff year if the query specifies a relative time window.

    Example: "last two years" in 2026 → 2024 (include 2024–2025, exclude 2023 and earlier).
    Returns None if no time window is detected, meaning no date filtering is applied.
    """
    match = re.search(r"(?:last|past)\s+(\w+|\d+)\s+year", query.lower())
    if not match:
        return None
    raw = match.group(1)
    n = int(raw) if raw.isdigit() else _YEAR_WORDS.get(raw)
    if n is None:
        return None
    return date.today().year - n


def _year_from_period(period: str) -> int | None:
    m = re.match(r"(\d{4})", period)
    return int(m.group(1)) if m else None


def detect_tickers(query: str) -> list[str]:
    """Find known ticker symbols or company names mentioned in the query."""
    query_upper = query.upper()
    found: set[str] = set()

    for ticker in KNOWN_TICKERS:
        if re.search(rf"\b{re.escape(ticker)}\b", query_upper):
            found.add(ticker)

    for name, ticker in NAME_TO_TICKER.items():
        if re.search(rf"\b{re.escape(name)}\b", query_upper):
            found.add(ticker)

    return list(found)


def retrieve(query: str, openai_client, collection) -> list[Chunk]:
    """Retrieve the top-k most relevant chunks from ChromaDB for a given query.

    Two key behaviors:

    1. Per-ticker balanced retrieval: when multiple companies are mentioned,
       a separate ChromaDB sub-query is fired per ticker. Without this, the
       company with the most filings dominates raw semantic similarity results
       and other companies get little or no coverage.

    2. Time-window filtering at retrieval: if the query contains "last N years",
       we fetch 3× more chunks than needed and discard those outside the window
       before anything reaches the LLM. A prompt-only instruction ("don't cite
       old data") is unreliable because the model still sees old chunks in its
       context and uses them. Filtering here ensures the LLM never sees
       out-of-window data in the first place.
    """
    tickers = detect_tickers(query)
    cutoff_year = parse_time_window(query)
    # Fetch extra chunks upfront to absorb losses from time-window filtering
    fetch_k = TOP_K * 3 if cutoff_year else TOP_K

    def embed(text: str) -> list[float]:
        resp = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=[text])
        return resp.data[0].embedding

    def make_chunk(doc: str, meta: dict) -> Chunk:
        return Chunk(
            text=doc,
            ticker=meta["ticker"],
            filing_type=meta["filing_type"],
            period=meta["period"],
        )

    def within_window(chunk: Chunk) -> bool:
        if cutoff_year is None:
            return True
        yr = _year_from_period(chunk.period)
        return yr is not None and yr >= cutoff_year

    if len(tickers) > 1:
        # Fire one sub-query per ticker so each company gets fair representation.
        # max(5, ...) ensures we always request a meaningful number even if there
        # are many tickers and fetch_k // len(tickers) would round to a tiny value.
        per_ticker_k = max(5, fetch_k // len(tickers))
        seen: dict[str, Chunk] = {}
        for ticker in tickers:
            results = collection.query(
                query_embeddings=[embed(query)],
                n_results=per_ticker_k,
                where={"ticker": ticker},
            )
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                # Deduplicate by first 80 chars to avoid near-identical chunks
                # from overlapping windows of the same filing passage.
                key = doc[:80]
                if key not in seen:
                    seen[key] = make_chunk(doc, meta)
        chunks = [c for c in seen.values() if within_window(c)]
        return chunks[:TOP_K]

    # Single-ticker or no-ticker path: one query with an optional ticker filter
    embedding = embed(query)
    kwargs: dict = {"query_embeddings": [embedding], "n_results": fetch_k}
    if len(tickers) == 1:
        kwargs["where"] = {"ticker": tickers[0]}
    results = collection.query(**kwargs)
    chunks = [
        make_chunk(doc, meta)
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]
    return [c for c in chunks if within_window(c)][:TOP_K]
