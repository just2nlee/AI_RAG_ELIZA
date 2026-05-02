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

_SYSTEM_PROMPT_TEMPLATE = """Today's date is {today}. You are a senior financial analyst at a consulting firm. \
You have been given excerpts from SEC filings (10-K annual reports and 10-Q \
quarterly reports) to answer a client's business question.

Your answer must:
- Open with a 2-3 sentence executive summary
- Provide a structured breakdown (by company if multi-company, by theme if thematic)
- Cite every claim with [TICKER FILING_TYPE PERIOD] inline
- Flag where data is limited or absent in the provided excerpts
- If the question references a time window (e.g. "last two years"), do NOT cite or use data from sources outside that window — calculate the window relative to today's date and exclude older periods entirely. Briefly note at the end which periods were omitted and why.
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
    allow_origins=["http://localhost:5173"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

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
    sources = [
        {"ticker": c.ticker, "filing_type": c.filing_type, "period": c.period}
        for c in chunks
    ]
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
        raise HTTPException(status_code=503, detail=f"Index not ready: {e}")

    openai_client = get_openai_client()
    chunks = retrieve(request.question, openai_client, collection)

    return StreamingResponse(
        stream_response(chunks, request.question),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
