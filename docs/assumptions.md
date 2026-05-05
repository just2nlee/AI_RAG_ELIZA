# Design Assumptions

## Build-time assumptions

1. **Local ChromaDB is sufficient** - 246 filings at demo scale does not require a cloud vector DB. ChromaDB persists to disk and loads fast enough for a live demo. A managed service (Pinecone, Weaviate) would be appropriate at production scale.

2. **Semantic search alone is adequate** - BM25 hybrid retrieval is not required for this corpus. The business questions are well-formed and embed meaningfully with `text-embedding-3-small`. Hybrid retrieval would improve precision for exact financial figures (e.g. specific dollar amounts) and would be a natural next step.

3. **No reranking step** - Top-15 cosine similarity results provide sufficient quality for this corpus size. A cross-encoder reranker (e.g. Cohere Rerank) could be added as a future improvement to boost precision on ambiguous queries.

4. **Multi-company detection is keyword-based** - Ticker symbols and plain-English company names in the query are detected via string matching against a known ticker list. No NER model is needed at this scale; the known-company universe is fixed and small.

5. **Index is pre-built before the demo** - `index.py` is run once offline. Cold-start indexing time (~10–15 minutes for 246 filings) is not part of the live query flow.

6. **Single question/answer paradigm** - The UI does not maintain conversation history. Each query is independent, which maps cleanly onto the single-LLM-call constraint.

7. **No authentication or rate limiting** - Demo-only system; no user auth, API key management, or production hardening required.

---

## Assumptions invalidated during build

These assumptions were made upfront but proved incorrect during testing. Documented here to show how the design evolved.

**8. Users will use ticker symbols in queries**
Original assumption: users would write "AAPL" or "TSLA." In practice, users write "Apple," "Tesla," "JPMorgan." Without a name→ticker mapping, `detect_tickers` returned nothing for natural language queries, the per-ticker sub-query logic never fired, and one company dominated the semantic search results.

Fix: added `NAME_TO_TICKER` dict in `retrieval.py` mapping common names and aliases to their tickers before the ChromaDB sub-queries run.

**9. A prompt instruction is sufficient to enforce time-window constraints**
Original assumption: telling the model "do not cite sources outside the last two years" would reliably exclude old data. In practice the model still used 2022/2023 chunks because they were semantically relevant and visible in its context window — the instruction competed with the in-context evidence and lost.

Fix: time-window filtering moved to the retrieval layer. `retrieve()` now detects the time window in the query, calculates a cutoff year, fetches 3× more chunks, and discards out-of-window chunks before anything reaches the LLM. See `docs/prompt_iterations.md` v2.1 for full rationale.

**10. The model can interpret relative time references without a date anchor**
Original assumption: phrases like "the last two years" would be interpreted reasonably. In practice the model had no reference point for "now" and produced inconsistent temporal reasoning.

Fix: today's date is injected into the system prompt at request time via `get_system_prompt()` in `backend/main.py`.
