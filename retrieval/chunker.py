"""
FinSight — Document Chunker
============================
Parses downloaded SEC 10-K HTML filings into clean, overlapping text
chunks with metadata attached — ready for embedding and storage in Qdrant.

Why chunking matters:
    Embedding models and LLMs have token limits — you can't feed an
    entire 10-K filing at once. Chunking breaks documents into pieces
    small enough to embed and retrieve individually.

Why overlap matters:
    A sliding window with overlap ensures that sentences near chunk
    boundaries aren't cut off mid-thought, preserving context across
    chunk edges.

Flow:
    .htm file → strip HTML → clean text → split into chunks → attach metadata
"""

import re
from pathlib import Path
from config import CHUNK_SIZE, CHUNK_OVERLAP

"""
    Converts a raw SEC .htm filing into plain text.

    Why not use a library like BeautifulSoup?
        We're deliberately keeping dependencies minimal.
        SEC filings follow a predictable structure, so regex stripping
        is reliable enough here without the overhead of a full parser.

    Steps:
        1. Remove <script> and <style> blocks entirely
        2. Replace block-level tags with newlines to preserve structure
        3. Strip all remaining HTML tags
        4. Decode common HTML entities (&amp; &nbsp; etc.)
    """

def parse_htm_file(filepath: Path) -> str:

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Step 1: Remove script and style blocks completely
    # re.DOTALL makes . match newlines too (blocks span multiple lines)
    content = re.sub(r"<script[^>]*>.*?</script>", " ", content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r"<style[^>]*>.*?</style>",  " ", content, flags=re.DOTALL | re.IGNORECASE)

    # Step 2: Replace block-level closing tags with newlines
    # This preserves paragraph breaks as whitespace in the plain text
    for tag in ["p", "div", "tr", "li", "h1", "h2", "h3", "h4", "br"]:
        content = re.sub(rf"</{tag}>", "\n", content, flags=re.IGNORECASE)

    # Step 3: Strip all remaining HTML tags
    content = re.sub(r"<[^>]+>", " ", content)

    # Step 4: Decode common HTML entities
    content = (content
        .replace("&amp;",  "&")
        .replace("&lt;",   "<")
        .replace("&gt;",   ">")
        .replace("&nbsp;", " ")
        .replace("&#160;", " ")
        .replace("&quot;", '"')
    )

    return content

    """
    Removes noise from extracted text.

    What we're cleaning:
        - Excessive whitespace and newlines (very common in parsed HTML)
        - Page numbers
        - Table separator lines (rows of dashes/underscores)
        - "Table of Contents" boilerplate
    """

def clean_text(text: str) -> str:
    
    # Collapse multiple spaces and newlines into single space
    text = re.sub(r"\s+", " ", text)

    # Remove page number patterns like "Page 45 of 120"
    text = re.sub(r"Page \d+ of \d+", "", text)

    # Remove table separator lines (e.g. "----------" or "__________")
    text = re.sub(r"[-_]{4,}", "", text)

    # Remove "Table of Contents" boilerplate
    text = re.sub(r"Table of Contents", "", text, flags=re.IGNORECASE)

    return text.strip()

    """
    Splits text into overlapping chunks using a sliding window.

    Why sentence-boundary splitting?
        Splitting mid-sentence produces chunks that start or end
        abruptly, which confuses the embedding model. We try to
        split at the nearest sentence boundary (. ! ?) near the
        end of each chunk window.

    Args:
        text: Clean plain text string

    Returns:
        List of text chunks, each between 100 and CHUNK_SIZE characters
    """

def chunk_text(text: str) -> list[str]:
    
    # If text is shorter than one chunk, return it as-is
    if len(text) <= CHUNK_SIZE:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE

        # If we've reached the end of the document, take the remainder
        if end >= len(text):
            chunks.append(text[start:].strip())
            break

        # Try to find a sentence boundary near the end of this window
        # Search backwards from 'end' up to 200 chars back
        split_at = end
        for i in range(end, max(start + CHUNK_SIZE // 2, end - 200), -1):
            if text[i] in ".!?":
                split_at = i + 1
                break

        chunk = text[start:split_at].strip()
        if chunk:
            chunks.append(chunk)

        # Slide the window forward, but back up by CHUNK_OVERLAP
        # so the next chunk shares some context with this one
        start = split_at - CHUNK_OVERLAP

    # Filter out very short chunks — they're usually noise
    return [c for c in chunks if len(c) > 100]

    """
    Full pipeline for one filing: parse → clean → chunk → tag with metadata.

    The metadata attached to each chunk is critical — it's what allows
    Qdrant to filter results by company or date during retrieval.
    Without metadata, you couldn't answer "search only Apple filings".

    Args:
        filepath    : Path to the .htm filing file
        ticker      : Stock ticker (e.g. "AAPL")
        company     : Full company name (e.g. "Apple Inc.")
        filing_type : e.g. "10-K"
        filing_date : e.g. "2024-11-01"

    Returns:
        List of chunk dicts, each with 'text' and 'metadata' keys
    """

def process_filing(
    filepath: Path,
    ticker: str,
    company: str,
    filing_type: str,
    filing_date: str,
) -> list[dict]:
    
    print(f"  Parsing   : {filepath.name}")

    # Parse HTML to plain text
    text = parse_htm_file(filepath)

    # Clean the extracted text
    text = clean_text(text)

    if not text:
        print(f"  Warning   : No text extracted from {filepath.name}")
        return []

    # Split into overlapping chunks
    chunks = chunk_text(text)
    print(f"  Chunks    : {len(chunks)} chunks from {len(text):,} characters")

    # Attach metadata to each chunk
    # chunk_index tells us the position of this chunk within the document
    # — useful for debugging retrieval later
    return [
        {
            "text": chunk,
            "metadata": {
                "ticker":      ticker.upper(),
                "company":     company,
                "filing_type": filing_type,
                "filing_date": filing_date,
                "chunk_index": i,
                "source_file": filepath.name,
            },
        }
        for i, chunk in enumerate(chunks)
    ]