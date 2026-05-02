# RAG SEC Filing Intelligence — Design Spec
**Date:** 2026-05-02  
**Role:** FDE interview assessment at AI consultancy startup  
**Timebox:** 4 hours build + 45-minute panel presentation with Q&A

---

## Overview

A retrieval-augmented generation (RAG) system that answers business questions about SEC financial filings. The user enters a natural-language question; the system retrieves relevant filing excerpts and produces a single, well-structured answer via one LLM API call.

**Hard constraint:** The final answer must come from exactly one LLM API call. Indexing and retrieval may run beforehand.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector store | ChromaDB (local, persistent) |
| Backend | FastAPI + Python |
| LLM (single call) | OpenAI `gpt-4o` with streaming |
| Frontend | React + Vite + TypeScript + shadcn/ui + Tailwind CSS |
| Documentation | `docs/prompt_iterations.md`, `docs/assumptions.md` |

---

## Architecture

### Three-phase pipeline

**Phase 1 — Indexing (run once, offline)**
- Script: `index.py`
- Reads all 246 `.txt` filings from `edgar_corpus/`
- Parses metadata (ticker, filing_type, period) from `manifest.json`
- Chunks each filing into 500-token chunks with 50-token overlap
- Embeds each chunk with `text-embedding-3-small`
- Persists to ChromaDB with metadata stored alongside each chunk

**Phase 2 — Retrieval (per query, before LLM call)**
- Module: `backend/retrieval.py`
- Embeds the user query with `text-embedding-3-small`
- Runs cosine similarity search against ChromaDB
- Default: top-15 chunks globally
- Multi-company queries: per-ticker sub-queries (one retrieval pass per detected company), results merged and deduplicated
- Optional metadata filtering by filing_type or date if explicitly referenced in the question

**Phase 3 — Generation (single LLM call)**
- Endpoint: `POST /query` in `backend/main.py`
- Assembles structured prompt with retrieved chunks (each labeled `[TICKER FILING_TYPE PERIOD]`)
- Calls `gpt-4o` with streaming
- Returns streamed answer + source citations to frontend

---

## File Structure

```
AI_RAG_ELIZA/
├── index.py                        # One-time indexing script
├── backend/
│   ├── main.py                     # FastAPI app, POST /query endpoint
│   ├── retrieval.py                # ChromaDB query logic
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   └── components/
│   └── package.json
├── docs/
│   ├── prompt_iterations.md        # Log of prompt versions (what changed, why)
│   ├── assumptions.md              # Documented assumptions
│   └── superpowers/specs/          # This file
├── edgar_corpus/                   # 246 SEC filings + manifest.json
├── .env                            # OPENAI_API_KEY
└── README.md
```

---

## Retrieval Design

| Parameter | Value | Rationale |
|---|---|---|
| Chunk size | 500 tokens | Rich enough context per chunk; small enough to stay focused |
| Overlap | 50 tokens | Prevents answer-critical sentences splitting across boundaries |
| Top-k | 15 chunks | Enough coverage for multi-company questions without bloating prompt |
| Multi-company | Per-ticker sub-queries, merged | Prevents one company dominating semantic similarity |
| Metadata | ticker, filing_type, period | Enables citation display; supports future filtering |

**Not implemented (documented as assumptions):**
- Reranking — adds latency/complexity for marginal gain in 4-hour window
- HyDE / query expansion — business questions embed well without it
- Hybrid BM25 + semantic — ChromaDB semantic search sufficient for this corpus size

---

## Prompt Design

**System prompt (static, cached across queries):**
```
You are a senior financial analyst at a consulting firm. You have been given excerpts 
from SEC filings (10-K annual reports and 10-Q quarterly reports) to answer a client's 
business question.

Your answer must:
- Open with a 2-3 sentence executive summary
- Provide a structured breakdown (by company if multi-company, by theme if thematic)
- Cite every claim with [TICKER FILING_TYPE PERIOD] inline
- Flag where data is limited or absent in the provided excerpts
- Be written for a C-suite audience: precise, professional, no filler

Answer only from the provided excerpts. Do not use prior knowledge about these companies.
```

**User prompt (assembled per query):**
```
FILING EXCERPTS:
[AAPL 10-K 2024-Q4] <chunk text>
[TSLA 10-Q 2024-Q3] <chunk text>
...

CLIENT QUESTION:
{user_question}
```

Prompt iteration log: every change is logged in `docs/prompt_iterations.md` with version, what changed, and why.

---

## Frontend Design

Inspired by eliza.com — enterprise, clean, minimal.

- **Header:** Logo left-aligned, white background, thin bottom border
- **Query panel:** Large prominent input + "Analyze" button; quick-example chips for demo
- **Answer panel:** Streams in below query; markdown-rendered with company headers; collapsible "Sources" section showing filing badges (ticker, type, date)
- **Loading state:** Input locks, subtle progress bar; GPT-4o thinking not exposed to user
- **Color palette:** White background, `#111` text, `#0066CC` accent blue
- **Components:** shadcn/ui throughout

---

## API

### `POST /query`
**Request:**
```json
{ "question": "What are the primary risk factors facing Apple, Tesla, and JPMorgan?" }
```
**Response (streaming):**
```json
{ "answer": "<streamed markdown>", "sources": [{"ticker": "AAPL", "filing_type": "10-K", "period": "2024-Q4"}] }
```

---

## Assumptions

1. Local ChromaDB sufficient — no cloud vector DB needed for 246 filings at demo scale
2. Semantic search alone adequate — BM25 hybrid not required
3. No reranking step — top-15 cosine similarity results are sufficient quality
4. Multi-company detection is keyword-based (ticker mentions in query)
5. Frontend runs on `localhost:5173`, backend on `localhost:8000`
6. Index is pre-built before the demo — cold-start indexing time not part of demo flow

---

## Quality Evaluation

Manual spot-checks against 3 example questions from the assessment:
1. Multi-company risk comparison (AAPL, TSLA, JPM) — verify citations map to correct filings
2. Single-company trend question (NVDA revenue) — verify temporal coverage across quarters
3. Thematic cross-company question (pharma regulatory risk) — verify breadth of companies covered

Criteria: citation accuracy, answer completeness, no hallucination of data not in corpus.
