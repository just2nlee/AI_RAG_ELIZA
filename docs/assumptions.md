# Assumptions

1. **Local ChromaDB is sufficient** — 246 filings at demo scale does not require a cloud vector DB. ChromaDB persists to disk and loads fast enough for a live demo.

2. **Semantic search alone is adequate** — BM25 hybrid retrieval not required for this corpus. The business questions are well-formed and embed meaningfully with `text-embedding-3-small`.

3. **No reranking step** — Top-15 cosine similarity results provide sufficient quality within the 4-hour build constraint. Cohere rerank or similar could be added as a future improvement.

4. **Multi-company detection is keyword-based** — Ticker symbols and company names mentioned in the query are detected via string matching against the known ticker list from `manifest.json`. No NER model is needed at this scale.

5. **Index is pre-built before the demo** — `index.py` is run once offline. Cold-start indexing time (~5-10 minutes for 246 filings) is not part of the live demo flow.

6. **Frontend on localhost:5173, backend on localhost:8000** — Both servers run locally on the presenter's machine. No deployment or cloud hosting required for the panel interview.

7. **Single question/answer paradigm** — The UI does not maintain chat history. Each query is independent, matching the assessment's single-LLM-call constraint cleanly.

8. **No authentication or rate limiting** — Demo-only system; no user auth, API key management, or production hardening required.

---

## Post-build fixes (2026-05-02)

9. **Time-window filtering must happen at retrieval, not in the prompt** — When the user asks "last two years," a prompt-only constraint ("do not cite old data") is unreliable because the model still sees 2022/2023 chunks in its context and uses them. The correct approach is to detect the time window in the query, calculate a cutoff year, fetch 3× more chunks from ChromaDB, and filter out out-of-window chunks before anything reaches the LLM. Implemented in `retrieve()` via `parse_time_window()`.

10. **Today's date must be injected into the system prompt** — Without a date anchor, the model cannot correctly interpret relative time references like "the last two years." The system prompt is now dynamic: `get_system_prompt()` in `backend/main.py` stamps the current date at request time so the model knows what "now" means.

11. **Anaconda Python must be used for all backend commands on this machine** — `python` and `python3` on this Windows machine point to non-functional Microsoft Store stubs. All backend commands (pip, uvicorn, pytest) must be run via `C:\Users\justi\anaconda3\python.exe` or `conda run`. Packages are installed into the Anaconda base environment.
