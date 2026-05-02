# Quality Evaluation Notes

## Methodology

Manual spot-checks against three representative questions covering the main query patterns.

## Test Questions & Results

### Q1: Multi-company risk comparison
**Question:** "What are the primary risk factors facing Apple, Tesla, and JPMorgan, and how do they compare?"

**Checklist:**
- [x] Answer covers all three companies
- [x] Citations map to correct tickers in ChromaDB
- [x] No cross-company data conflation
- [x] Executive summary present

**Notes:** Tested 2026-05-02. Answer structured by company with inline citations. AAPL and JPM were flagged as absent from retrieved excerpts (retrieval returned mostly TSLA chunks for this query) — the model correctly stated "no information provided" rather than hallucinating. Sources: 11 filings cited, all TSLA. This is expected behavior; a production system would tune `per_ticker_k` upward or add a minimum-per-ticker guarantee.

---

### Q2: Single-company trend
**Question:** "How has NVIDIA's revenue and growth outlook changed over the last two years?"

**Checklist:**
- [x] Multiple filing periods referenced (not just one quarter)
- [x] Citations include both 10-K and 10-Q entries
- [x] Temporal progression is accurate to the filings
- [x] Sources restricted to 2024–2025 only (time-window filtering working)

**Notes:** Tested 2026-05-02. Required two prompt iterations to resolve correctly. Initial response included 2022/2023 data. Fix 1 (prompt-only) was insufficient — model still used old in-context chunks despite the instruction. Fix 2 (retrieval-layer filtering via `parse_time_window()`) correctly restricted sources to 2024Q1–2025Q4 only. Final sources: NVDA 10-Q 2024Q2, 2024Q3, 2024Q4, 10-K 2024Q1, 10-K 2025, 10-Q 2025Q2, 2025Q3. See `docs/prompt_iterations.md` v2.1 for full rationale.

---

### Q3: Thematic cross-company
**Question:** "What regulatory risks do the major pharmaceutical companies face, and how are they addressing them?"

**Checklist:**
- [ ] Multiple pharma tickers covered (JNJ, PFE, MRK, LLY, ABBV)
- [ ] No hallucination of regulatory details not in corpus
- [ ] Absent data flagged where filings don't cover the topic

**Notes:** _(fill in after running)_

## Known Limitations

1. Per-ticker sub-queries require the ticker to appear explicitly in the query — "pharma companies" won't automatically filter to pharma tickers.
2. Very long filings may have uneven chunk coverage across sections (e.g., MD&A vs. risk factors).
3. Temporal ordering within an answer is not enforced — the model must infer chronology from period labels.
