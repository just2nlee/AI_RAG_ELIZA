from __future__ import annotations

import json
import os
import sys
from datetime import date
from typing import Generator

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import OpenAI
from pydantic import BaseModel

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

sys.path.insert(0, os.path.dirname(__file__))
from retrieval import Chunk, format_chunks, retrieve

# Today's date is injected at request time so the model can correctly interpret
# relative time references like "the last two years". A static date in the
# prompt would drift and produce wrong temporal reasoning over time.
_SYSTEM_PROMPT_TEMPLATE = """Today's date is {today}. You are a senior financial analyst at a consulting firm. \
You have been given excerpts from SEC filings (10-K annual reports and 10-Q \
quarterly reports) to answer a client's business question.

Your answer must:
- Open with a `## Executive Summary` section (2-3 sentences)
- Use `##` markdown headers for major sections and `###` for subsections (e.g. individual companies or segments)
- Use bullet points for lists of data points within sections
- Cite every claim with [TICKER FILING_TYPE PERIOD] inline
- Flag where data is limited or absent in the provided excerpts
- If the question references a time window (e.g. "last two years"), do NOT cite or use data from sources outside that window — calculate the window relative to today's date and exclude older periods entirely.
- Be written for a C-suite audience: precise, professional, no filler

Do NOT:
- Speculate beyond what is stated in the provided excerpts
- Combine financial figures across filing periods without explicitly flagging the aggregation
- Reference company information from outside the provided excerpts
- Infer data for a company that has no relevant excerpts — state the absence explicitly

Answer only from the provided excerpts."""


def get_system_prompt() -> str:
    return _SYSTEM_PROMPT_TEMPLATE.format(today=date.today().strftime("%B %d, %Y"))


app = FastAPI(title="SEC RAG API")
app.add_middleware(
    CORSMiddleware,
    # Restrict to localhost only — this is a local demo, not a public API.
    allow_origins=["http://localhost:5173"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Module-level singletons: OpenAI client and ChromaDB collection are initialized
# once on first use and reused across requests to avoid reconnection overhead.
_openai_client: OpenAI | None = None
_collection = None


def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def get_collection():
    global _collection
    if _collection is None:
        chroma_path = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
        client = chromadb.PersistentClient(path=chroma_path)
        _collection = client.get_collection("sec_filings")
    return _collection


class QueryRequest(BaseModel):
    question: str


def stream_response(chunks: list[Chunk], question: str) -> Generator[str, None, None]:
    """Yield SSE events: sources first, then streamed text deltas, then done.

    Sending sources before the text stream lets the frontend render source badges
    immediately while the answer is still generating, improving perceived latency.
    """
    sources = [
        {"ticker": c.ticker, "filing_type": c.filing_type, "period": c.period}
        for c in chunks
    ]
    # Multiple chunks from the same filing produce duplicate source entries;
    # deduplicate so the UI shows one badge per filing, not one per chunk.
    seen = set()
    unique_sources = []
    for s in sources:
        key = (s["ticker"], s["filing_type"], s["period"])
        if key not in seen:
            seen.add(key)
            unique_sources.append(s)

    yield f"data: {json.dumps({'type': 'sources', 'sources': unique_sources})}\n\n"

    user_prompt = (
        f"FILING EXCERPTS:\n{format_chunks(chunks)}\n\nCLIENT QUESTION:\n{question}"
    )
    openai_client = get_openai_client()

    # temperature=0 for deterministic, factual answers — financial data should
    # not vary between runs of the same query.
    stream = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": user_prompt},
        ],
        stream=True,
        temperature=0,
    )
    for event in stream:
        delta = event.choices[0].delta.content
        if delta:
            yield f"data: {json.dumps({'type': 'text', 'content': delta})}\n\n"

    yield f"data: {json.dumps({'type': 'done'})}\n\n"


@app.post("/query")
async def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        collection = get_collection()
    except Exception as e:
        # Surface a clear error if index.py hasn't been run yet
        raise HTTPException(status_code=503, detail=f"Index not ready: {e}")

    openai_client = get_openai_client()
    chunks = retrieve(request.question, openai_client, collection)

    # X-Accel-Buffering: no tells nginx not to buffer the SSE stream if this
    # is ever deployed behind a reverse proxy.
    return StreamingResponse(
        stream_response(chunks, request.question),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
