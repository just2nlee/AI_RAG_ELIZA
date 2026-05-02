import sys
sys.path.insert(0, "backend")
from retrieval import detect_tickers, format_chunks, Chunk

def test_detect_single_ticker():
    tickers = detect_tickers("What are AAPL's risk factors?")
    assert "AAPL" in tickers

def test_detect_multiple_tickers():
    tickers = detect_tickers("Compare AAPL, TSLA, and JPM risk factors")
    assert set(tickers) >= {"AAPL", "TSLA", "JPM"}

def test_detect_no_tickers():
    tickers = detect_tickers("What are the main risks facing pharma companies?")
    assert isinstance(tickers, list)

def test_detect_ticker_case_insensitive():
    tickers = detect_tickers("what does NVDA say about revenue?")
    assert "NVDA" in tickers

def test_format_chunks_labels():
    chunks = [
        Chunk(text="Revenue grew 12%.", ticker="AAPL", filing_type="10-K", period="2024Q3"),
        Chunk(text="Risks include competition.", ticker="MSFT", filing_type="10-Q", period="2023Q2"),
    ]
    result = format_chunks(chunks)
    assert "[AAPL 10-K 2024Q3]" in result
    assert "[MSFT 10-Q 2023Q2]" in result
    assert "Revenue grew 12%" in result
    assert "---" in result

def test_format_chunks_empty():
    result = format_chunks([])
    assert result == ""
