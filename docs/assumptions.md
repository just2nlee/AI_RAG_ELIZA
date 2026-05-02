# Assumptions

1. **Local ChromaDB is sufficient** — 246 filings at demo scale does not require a cloud vector DB. ChromaDB persists to disk and loads fast enough for a live demo.

2. **Semantic search alone is adequate** — BM25 hybrid retrieval not required for this corpus. The business questions are well-formed and embed meaningfully with `text-embedding-3-small`.

3. **No reranking step** — Top-15 cosine similarity results provide sufficient quality within the 4-hour build constraint. Cohere rerank or similar could be added as a future improvement.

4. **Multi-company detection is keyword-based** — Ticker symbols and company names mentioned in the query are detected via string matching against the known ticker list from `manifest.json`. No NER model is needed at this scale.

5. **Index is pre-built before the demo** — `index.py` is run once offline. Cold-start indexing time (~5-10 minutes for 246 filings) is not part of the live demo flow.

6. **Frontend on localhost:5173, backend on localhost:8000** — Both servers run locally on the presenter's machine. No deployment or cloud hosting required for the panel interview.

7. **Single question/answer paradigm** — The UI does not maintain chat history. Each query is independent, matching the assessment's single-LLM-call constraint cleanly.

8. **No authentication or rate limiting** — Demo-only system; no user auth, API key management, or production hardening required.
