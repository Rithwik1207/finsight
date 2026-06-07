"""
FinSight — SEC Filing Downloader
=================================
This script downloads 10-K annual report filings from the SEC EDGAR
public database for the 5 companies in our corpus.

What is a 10-K?
    A 10-K is a company's annual financial report filed with the SEC.
    It contains business overview, risk factors, revenue breakdown,
    competition analysis, and future outlook — making it the richest
    source of financial text for our RAG pipeline.

What is SEC EDGAR?
    EDGAR is the SEC's public database of all company filings.
    It exposes a free JSON API — no authentication required.
    Every company has a unique CIK (Central Index Key) identifier.

How this script works:
    1. For each company, hit the EDGAR API with their CIK
    2. Parse the JSON response to find all 10-K filings
    3. Download the last 2 years of 10-K filings per company
    4. Save each filing as an .htm file in data/filings/

Output:
    data/filings/AAPL_10-K_2024-11-01.htm
    data/filings/MSFT_10-K_2024-07-30.htm
    ... (10 files total, 2 per company)

Usage:
    python data/download_filings.py
"""

import time
import requests
from pathlib import Path


# ─────────────────────────────────────────────
# REQUEST HEADERS
# ─────────────────────────────────────────────
# SEC EDGAR requires a User-Agent header identifying who is making
# the request. Without this, SEC will block your requests (HTTP 403).
# Format: "AppName ContactName email@example.com"
HEADERS = {
    "User-Agent": "sairithwikgundu@gmail.com",  # replace with your email
}


# ─────────────────────────────────────────────
# OUTPUT DIRECTORY
# ─────────────────────────────────────────────
# All downloaded .htm files will be saved here.
# mkdir(parents=True, exist_ok=True) creates the folder if it doesn't
# exist, and silently does nothing if it already does.
SAVE_DIR = Path("data/filings")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# COMPANY LIST
# ─────────────────────────────────────────────
# The 5 Big Tech companies we're building our corpus from.
# CIK (Central Index Key) is the unique SEC identifier for each company.
# To add more companies, find their CIK at: https://www.sec.gov/cgi-bin/browse-edgar

COMPANIES = {
    "AAPL":  {"cik": "0000320193", "name": "Apple"},
    "MSFT":  {"cik": "0000789019", "name": "Microsoft"},
    "GOOGL": {"cik": "0001652044", "name": "Alphabet"},
    "NVDA":  {"cik": "0001045810", "name": "NVIDIA"},
    "META":  {"cik": "0001326801", "name": "Meta"},
}

# We only grab the last 2 annual filings per company.
# 2 years x 5 companies = 10 files total — enough for a rich index
# without downloading gigabytes of data.

FILING_TYPE = "10-K"
MAX_FILINGS_PER_COMPANY = 2


# ─────────────────────────────────────────────
# FUNCTION: GET FILINGS LIST
# ─────────────────────────────────────────────
"""
    Fetches the list of 10-K filings for a company from SEC EDGAR.

    How EDGAR's API works:
        Hitting https://data.sec.gov/submissions/CIK{cik}.json returns
        a JSON object with a 'filings.recent' key containing parallel
        arrays — forms[], accessionNumbers[], filingDates[], etc.
        Index 0 across all arrays refers to the same filing.

    Args:
        cik: The company's SEC CIK number (e.g. "0000320193" for Apple)

    Returns:
        List of dicts, each containing accession_number, filing_date,
        and primary_document for one 10-K filing.
    """
def get_filings_list(cik: str) -> list[dict]:
    
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    response = requests.get(url, headers=HEADERS)

    # raise_for_status() throws an exception immediately if the request
    # failed (e.g. network error, SEC blocked us) rather than continuing
    # with broken/empty data silently
    response.raise_for_status()
    data = response.json()

    # EDGAR returns parallel arrays — each index represents one filing
    filings = data["filings"]["recent"]
    forms            = filings["form"]
    accession_numbers = filings["accessionNumber"]
    filing_dates     = filings["filingDate"]
    primary_docs     = filings["primaryDocument"]

    results = []
    for i, form in enumerate(forms):
        # Only collect 10-K filings, stop once we have enough
        if form == FILING_TYPE and len(results) < MAX_FILINGS_PER_COMPANY:
            results.append({
                # EDGAR accession numbers look like "0000320193-24-000123"
                # but the download URL needs them without dashes
                "accession_number": accession_numbers[i].replace("-", ""),
                "filing_date": filing_dates[i],
                "primary_document": primary_docs[i],
            })

    return results


# ─────────────────────────────────────────────
# FUNCTION: DOWNLOAD A SINGLE FILING
# ─────────────────────────────────────────────
"""
    Downloads a single 10-K filing document and saves it to disk.

    EDGAR document URL format:
        https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{filename}
        Note: CIK here needs leading zeros stripped (int() does this)

    Args:
        cik             : Company CIK number
        accession_number: Filing accession number with dashes removed
        primary_doc     : Filename of the primary document (e.g. aapl-20240928.htm)
        save_path       : Where to save the file on disk

    Returns:
        True if download succeeded, False if it failed
    """
def download_filing(cik: str, accession_number: str, primary_doc: str, save_path: Path) -> bool:
    
    url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{int(cik)}/{accession_number}/{primary_doc}"
        # int(cik) strips leading zeros — URL needs "320193" not "0000320193"
    )

    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()

        # "wb" = write binary mode — important for HTML files that may
        # contain special characters or encoded content
        with open(save_path, "wb") as f:
            f.write(response.content)

        print(f"  Downloaded : {save_path.name} ({len(response.content) // 1024} KB)")
        return True

    except Exception as e:
        print(f"  Failed     : {e}")
        return False


# ─────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────
def download_all():
    """
    Orchestrates the full download process for all companies.
    Skips files that already exist so the script is safe to re-run.
    """
    print("Downloading 10-K filings from SEC EDGAR...\n")

    for ticker, info in COMPANIES.items():
        print(f"[{ticker}] {info['name']}")

        filings = get_filings_list(info["cik"])

        for filing in filings:
            # Standardized filename format: TICKER_10-K_YYYY-MM-DD.htm
            # This format is important — build_index.py parses the ticker
            # and date directly from this filename
            filename = f"{ticker}_10-K_{filing['filing_date']}.htm"
            save_path = SAVE_DIR / filename

            # Skip if already downloaded — safe to re-run the script
            if save_path.exists():
                print(f"  Already exists: {filename}")
                continue

            download_filing(
                cik=info["cik"],
                accession_number=filing["accession_number"],
                primary_doc=filing["primary_document"],
                save_path=save_path,
            )

            # SEC rate limit: be polite, pause between requests
            # SEC allows max 10 requests/second — 0.5s keeps us well within that
            time.sleep(0.5)

        print()


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
# This block only runs when you execute this file directly.
# It won't run if another file imports a function from here.
if __name__ == "__main__":
    download_all()