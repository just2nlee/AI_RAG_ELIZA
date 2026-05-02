# SEC Filing Intelligence — RAG Demo

A retrieval-augmented generation (RAG) system that answers business questions about SEC financial filings via a single GPT-4o API call.

## Setup

### Prerequisites
- Python 3.9+ (Anaconda recommended on Windows)
- Node.js 18+
- OpenAI API key

### 1. Environment
```bash
cp backend/.env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 2. Build the index (run once, ~15-20 minutes)
```bash
pip install -r backend/requirements.txt
python index.py
```

### 3. Start the backend
```bash
cd backend
uvicorn main:app --reload --port 8000
```

### 4. Start the frontend
```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

## Architecture

**Indexing (offline):** `index.py` chunks 246 SEC filings into 500-token segments with 50-token overlap, embeds them with `text-embedding-3-small`, and persists to ChromaDB locally.

**Retrieval (per query):** The backend detects company ticker symbols in the query, runs per-ticker sub-queries against ChromaDB to ensure balanced multi-company coverage, and selects the top-15 most relevant chunks.

**Generation (single LLM call):** Retrieved chunks are injected into a structured prompt with `[TICKER FILING_TYPE PERIOD]` labels. GPT-4o streams the response back to the frontend with citations.

## Dataset
246 filings (89 10-K, 157 10-Q) from 54 US public companies, 2022–2026.

## Running Tests
```bash
pytest tests/ -v
```

## Key Design Decisions
See `docs/assumptions.md` for documented assumptions and `docs/prompt_iterations.md` for prompt iteration history.
