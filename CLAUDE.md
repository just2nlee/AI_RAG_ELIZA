# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI engineering assessment: a **Retrieval-Augmented Generation (RAG) system** that answers business questions about SEC financial filings. The final answer must be produced in **exactly one LLM API call** — indexing and retrieval may happen beforehand.

## Dataset

**`edgar_corpus/`** — 246 SEC filings (89 10-K annual + 157 10-Q quarterly) from 54 major US public companies (AAPL, NVDA, MSFT, AMZN, JPM, etc.), covering 2022–2026. All files are plain `.txt`. Metadata is in `edgar_corpus/manifest.json`.

## Required Deliverables

- Indexing and retrieval code
- Single-call LLM integration
- Frontend UI (web interface)
- Log of prompt iterations (what changed, why)
- Final prompt template
- README with setup/run instructions
- Notes on quality evaluation

## Architecture

The pipeline has three phases:

1. **Indexing (offline)** — Parse filings → chunk text → embed chunks → store in vector DB
2. **Retrieval (per query)** — Embed query → similarity search → fetch top-k passages with metadata (ticker, filing type, date)
3. **Generation (single LLM call)** — Inject retrieved context into a structured prompt → return answer with citations

## Commands

> Commands will be documented here once the tech stack is chosen and `requirements.txt` / `package.json` are created.

Likely shape (Python stack):
```bash
python index.py          # Build/refresh vector index from edgar_corpus/
python app.py            # Start backend server
# or
streamlit run app.py     # If using Streamlit frontend
```

## Key Constraint

The final answer to any user query must come from **one LLM API call**. Multi-step chains or iterative LLM calls during inference are not allowed. All retrieval and context assembly must happen before that single call.

## Environment Variables

Document required env vars in a `.env.example` when the stack is finalized. Expected keys: `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`, plus any vector DB credentials.
