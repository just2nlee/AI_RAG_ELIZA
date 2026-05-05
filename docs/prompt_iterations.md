# Prompt Iteration Log

## v1 — Initial prompt (2026-05-02)

**System prompt:**
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

**User prompt template:**
```
FILING EXCERPTS:
{formatted_chunks}

CLIENT QUESTION:
{user_question}
```

**Rationale:** Baseline prompt establishing an analyst persona, citation format, and a grounding constraint. Positive instructions only — tells the model what to do, not what to avoid.

---

## v2 — Date injection + temporal window flag (2026-05-02)

**What changed:**
- System prompt is now dynamic - today's date is injected at request time (`Today's date is {today}.`)
- Added instruction: "If the question references a time window (e.g. 'last two years'), explicitly note any cited sources that fall outside that window"
- Added negative constraints: do not speculate, do not combine figures across periods without flagging, do not infer data for companies with no relevant excerpts

**Why:** Without a date anchor, the model has no basis to interpret relative time references like "the last two years." It would include 2023 data in a "last two years" query even though today is May 2026. Injecting the current date gives the model the context to reason about the time window. The temporal flagging instruction tells it to surface out-of-window sources explicitly rather than silently including stale data.

**Outcome:** Improved citation transparency but the model still used out-of-window chunks in its answers. The instruction competed with the in-context evidence and lost. Led directly to v2.1.

---

## v2.1 — Retrieval-layer time filtering (2026-05-02)

**What changed:**
- Prompt instruction updated from "flag out-of-window sources" to "do NOT cite or use data from sources outside that window"
- `retrieve()` in `retrieval.py` now detects time-window phrases ("last N years", "past N years") in the query, calculates the cutoff year relative to today, fetches 3× more chunks from ChromaDB, and filters out chunks whose period falls before the cutoff — before anything reaches the LLM

**Why:** Prompt-only temporal constraints are unreliable. Even with a strong "do not use old data" instruction, the model still sees 2022/2023 chunks in its context window and uses them because they are semantically relevant to the question. The instruction competes with the in-context evidence and loses.

The correct fix is to enforce the time window at the retrieval layer so that out-of-window chunks are never passed to the model in the first place. This reflects a general principle: use the prompt to shape *how* the model answers, but use retrieval filters to control *what data* it sees. Trying to do both jobs with the prompt alone leads to inconsistent behavior.

**Outcome:** "Last two years" from May 2026 correctly surfaces only 2024–2025 filings. Out-of-window data no longer appears in answers regardless of semantic relevance.

---

## Final prompt (live)

**System prompt:**
```
Today's date is {today}. You are a senior financial analyst at a consulting firm.
You have been given excerpts from SEC filings (10-K annual reports and 10-Q
quarterly reports) to answer a client's business question.

Your answer must:
- Open with a ## Executive Summary section (2-3 sentences)
- Use ## markdown headers for major sections and ### for subsections
- Use bullet points for lists of data points within sections
- Cite every claim with [TICKER FILING_TYPE PERIOD] inline
- Flag where data is limited or absent in the provided excerpts
- If the question references a time window (e.g. "last two years"), do NOT cite or
  use data from sources outside that window — calculate the window relative to
  today's date and exclude older periods entirely
- Be written for a C-suite audience: precise, professional, no filler

Do NOT:
- Speculate beyond what is stated in the provided excerpts
- Combine financial figures across filing periods without explicitly flagging the aggregation
- Reference company information from outside the provided excerpts
- Infer data for a company that has no relevant excerpts — state the absence explicitly

Answer only from the provided excerpts.
```

**User prompt template:**
```
FILING EXCERPTS:
{formatted_chunks}

CLIENT QUESTION:
{user_question}
```
