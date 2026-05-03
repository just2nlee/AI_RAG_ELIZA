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

**Notes:** Tested 2026-05-02. Initially failed — retrieval returned only TSLA chunks because `detect_tickers` only matched exact ticker symbols, so "Apple", "Tesla", "JPMorgan" returned no matches, the per-ticker sub-query logic never fired, and TSLA dominated by raw semantic similarity. Fixed by adding `NAME_TO_TICKER` mapping in `retrieval.py`. After fix, all three companies appear in sources and the answer provides balanced coverage. See `docs/assumptions.md` assumption #9.

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

### Q4: Direct competitor comparison
**Question:** "How are NVIDIA and AMD positioning themselves in the AI chip market, and what risks do they face?"

**Checklist:**
- [x] Both companies detected by name ("NVIDIA" → NVDA, "AMD" → AMD)
- [x] Per-ticker sub-queries fire for both tickers
- [x] NVDA well covered (5 sources: 10-K 2025, 10-K 2024Q1, 10-Q 2024Q2, 10-K 2022Q1, 10-Q 2023Q2)
- [ ] AMD coverage is thin — only 1 source (10-K 2026)
- [ ] Older NVDA sources (2022Q1, 2023Q2) included since no time window is specified in the query

**Notes:** Tested 2026-05-03. The thin AMD coverage is a data gap — AMD has only 1 filing in the corpus vs. 16 for NVDA. The per-ticker logic gives AMD `max(5, 15//2) = 7` chunks to retrieve, but with only 1 filing there's limited material. This is expected behavior given the dataset. The inclusion of older NVDA sources (2022, 2023) is correct — the query doesn't specify a time window, so no filtering is applied. A production system could add a default recency bias even without an explicit time window.

---

### Q5: Cross-institution financial thematic
**Question:** "What do JPMorgan, Goldman Sachs, and Bank of America disclose about their exposure to interest rate risk?"

**Checklist:**
- [x] All three institutions detected by name ("JPMorgan" → JPM, "Goldman Sachs" → GS, "Bank of America" → BAC)
- [x] Per-ticker sub-queries fire for all three tickers
- [x] JPM well covered (4 sources: 10-K 2026, 10-Q 2025Q1, 2025Q2, 2025Q3)
- [x] BAC well covered (3 sources: 10-K 2025, 10-Q 2025Q1, 2025Q3)
- [ ] GS coverage thin — only 1 source (10-K 2025)
- [x] All sources naturally recent (2025–2026) — no time window filtering needed

**Notes:** Tested 2026-05-03. Strong result for JPM and BAC. GS thinness is again a corpus data gap (1 filing). The answer correctly covers all three institutions and the model appropriately flags where GS detail is limited. This question demonstrates the system's strength for cross-institution thematic queries where the corpus is well-populated.

---

## Known Limitations

1. Per-ticker sub-queries require the ticker or a known company name — generic sector terms like "pharma companies" or "big banks" won't map to specific tickers.
2. Corpus coverage is uneven — some companies have 1 filing (AMD, GS) while others have 16+ (NVDA, TSLA). Thin-corpus companies will always produce weaker answers regardless of retrieval quality.
3. Very long filings may have uneven chunk coverage across sections (e.g., MD&A vs. risk factors).
4. Temporal ordering within an answer is not enforced — the model infers chronology from period labels.
5. No default recency bias — queries without an explicit time window may surface older filings if they score higher on semantic similarity.
