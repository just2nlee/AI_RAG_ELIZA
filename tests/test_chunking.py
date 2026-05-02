import sys
sys.path.insert(0, "backend")
from retrieval import parse_filename, chunk_text

def test_parse_filename_with_period():
    result = parse_filename("AAPL_10K_2022Q3_2022-10-28_full.txt")
    assert result == {"ticker": "AAPL", "filing_type": "10-K", "period": "2022Q3"}

def test_parse_filename_without_period():
    result = parse_filename("GS_10K_2025-02-27_full.txt")
    assert result == {"ticker": "GS", "filing_type": "10-K", "period": "2025"}

def test_parse_filename_10q():
    result = parse_filename("JPM_10Q_2025Q1_2025-05-01_full.txt")
    assert result == {"ticker": "JPM", "filing_type": "10-Q", "period": "2025Q1"}

def test_chunk_text_splits_by_tokens():
    text = "word " * 600
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) >= 2

def test_chunk_text_short_text_single_chunk():
    text = "hello world"
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) == 1
    assert "hello" in chunks[0]

def test_chunk_text_overlap_produces_shared_content():
    text = "word " * 600
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) >= 2
