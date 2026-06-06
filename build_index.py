"""
FinSight — Index Builder
=========================
This is the factory manager script that runs the full ingestion pipeline.
Run this once after downloading filings to populate your Qdrant collection.

What it does:
    1. Scans data/filings/ for all downloaded .htm filing files
    2. Parses the ticker, filing type, and date from each filename
    3. Passes each file through the chunker (parse → clean → chunk)
    4. Passes the chunks to the vector store (embed → store in Qdrant)
    5. Prints a summary of how many chunks were stored

Prerequisites:
    - .env file with QDRANT_URL and QDRANT_API_KEY filled in
    - 10-K files downloaded in data/filings/ (run download_filings.py first)

Usage:
    python build_index.py
"""

import re
from pathlib import Path
from retrieval.chunker import process_filing
from retrieval.vector_store import VectorStore


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

# Folder where downloaded .htm filing files live
FILINGS_DIR = Path("data/filings")

# Filename pattern we expect: TICKER_10-K_YYYY-MM-DD.htm
# We use regex to extract ticker, filing type, and date from the name
# Example: "AAPL_10-K_2024-11-01.htm"
#           ^^^^  ^^^^  ^^^^^^^^^^
#           ticker type   date
FILENAME_PATTERN = re.compile(
    r"^(?P<ticker>[A-Z]+)_(?P<filing_type>[^_]+)_(?P<date>\d{4}-\d{2}-\d{2})\."
)

# Map each ticker to its full company name
# Used when attaching metadata to chunks
COMPANY_NAMES = {
    "AAPL":  "Apple Inc.",
    "MSFT":  "Microsoft Corporation",
    "GOOGL": "Alphabet Inc.",
    "NVDA":  "NVIDIA Corporation",
    "META":  "Meta Platforms Inc.",
}


# ─────────────────────────────────────────────
# HELPER: PARSE FILENAME
# ─────────────────────────────────────────────
"""
    Extracts ticker, filing type, and date from a filename.

    Why parse from filename instead of opening the file?
        It's faster — we get the metadata without reading the file.
        The download script saved files with this exact format
        precisely so we can do this here.

    Returns:
        Dict with ticker, filing_type, filing_date
        None if filename doesn't match expected pattern
    """

def parse_filename(filename: str) -> dict | None:
    
    match = FILENAME_PATTERN.match(filename)
    if not match:
        return None
    return {
        "ticker":       match.group("ticker"),
        "filing_type":  match.group("filing_type"),
        "filing_date":  match.group("date"),
    }


# ─────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────

def build_index():
    print("=" * 50)
    print("  FinSight — Index Builder")
    print("=" * 50)

    # Step 1: Connect to Qdrant
    # This also creates the collection if it doesn't exist yet
    print("\nConnecting to Qdrant...")
    store = VectorStore()
    print("Connected.\n")

    # Step 2: Find all .htm filing files in the filings folder
    filing_files = list(FILINGS_DIR.glob("*.htm"))

    if not filing_files:
        print(f"No files found in {FILINGS_DIR}/")
        print("Run python data/download_filings.py first.")
        return

    print(f"Found {len(filing_files)} filing(s) to process.\n")

    total_chunks = 0

    # Step 3: Process each file one by one
    for filepath in filing_files:

        # Extract metadata from the filename
        meta = parse_filename(filepath.name)

        if not meta:
            print(f"Skipping (unrecognized filename): {filepath.name}")
            continue

        ticker  = meta["ticker"]
        company = COMPANY_NAMES.get(ticker, ticker)

        print(f"[{ticker}] {company} — {meta['filing_type']} {meta['filing_date']}")

        # Step 4: Chunk the filing
        # process_filing returns a list of dicts: [{text, metadata}, ...]
        chunks = process_filing(
            filepath=filepath,
            ticker=ticker,
            company=company,
            filing_type=meta["filing_type"],
            filing_date=meta["filing_date"],
        )

        if not chunks:
            print("  No chunks extracted, skipping.\n")
            continue

        # Step 5: Embed and store chunks in Qdrant
        print(f"  Upserting  : {len(chunks)} chunks to Qdrant...")
        count = store.upsert_documents(chunks)
        total_chunks += count
        print(f"  Done.\n")

    # Step 6: Print final summary
    print("=" * 50)
    print(f"  Index build complete.")
    print(f"  Total chunks stored : {total_chunks}")

    # Confirm with Qdrant how many vectors are now in the collection
    info = store.get_collection_info()
    print(f"  Vectors in Qdrant   : {info['total_vectors']}")
    print("=" * 50)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    build_index()