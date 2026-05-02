# Quality Evaluation Notes

## Methodology

Manual spot-checks against three representative questions covering the main query patterns.

## Test Questions & Results

### Q1: Multi-company risk comparison
**Question:** "What are the primary risk factors facing Apple, Tesla, and JPMorgan, and how do they compare?"

**Checklist:**
- [ ] Answer covers all three companies
- [ ] Citations map to correct tickers in ChromaDB
- [ ] No cross-company data conflation
- [ ] Executive summary present

**Notes:** _(fill in after running)_

---

### Q2: Single-company trend
**Question:** "How has NVIDIA's revenue and growth outlook changed over the last two years?"

**Checklist:**
- [ ] Multiple filing periods referenced (not just one quarter)
- [ ] Citations include both 10-K and 10-Q entries
- [ ] Temporal progression is accurate to the filings

**Notes:** _(fill in after running)_

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
