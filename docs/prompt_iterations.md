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

**Rationale:** Baseline prompt establishing analyst persona, citation format, and grounding constraint. Positive instructions only.

---

## v2 — Date injection + temporal window flag (2026-05-02)

**What changed:**
- System prompt is now dynamic — today's date is injected at request time (`Today's date is {today}.`)
- Added instruction: "If the question references a time window (e.g. 'last two years'), explicitly note any cited sources that fall outside that window"
- Added remaining negative constraints from v1 plan

**Why:** Without a date anchor, the model had no basis to interpret relative time references like "the last two years." It would include 2023 data in a "last two years" query even though today is May 2026. Injecting the current date gives the model the context to flag out-of-window sources. The temporal flagging instruction tells it to surface that information explicitly rather than silently including stale data.

**Status:** Implemented and live.

---

## v3 — Planned: add negative constraints (original v2 plan)

**What to change:** Add explicit out-of-scope / "do not" instructions to the system prompt.

**Candidates:**
- Do not speculate beyond what is stated in the provided excerpts
- Do not combine financial figures across filing periods without explicitly flagging the aggregation
- Do not reference company information from outside the provided corpus
- If a company is mentioned in the question but has no relevant excerpts, state that explicitly rather than inferring

**Why:** Negative constraints tighten answer quality by reducing hallucination at the edges — particularly for multi-period trend questions where the model may be tempted to interpolate.

**Status:** Pending — implement and evaluate against baseline after v1 is working end-to-end.
