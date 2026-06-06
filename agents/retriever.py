"""
FinSight — Retriever Agent
===========================
The second agent in the pipeline. Takes the search sub-tasks from
the Planner and executes each one against the Qdrant vector store.

What it does:
    1. Loops through each sub-task in state.plan
    2. Detects if a specific company ticker is mentioned
    3. Searches Qdrant with optional ticker filter
    4. Accumulates all retrieved chunks into state.chunks

Why loop through sub-tasks instead of one big search?
    Each sub-task is targeted at a specific aspect or company.
    Running them separately and accumulating results gives much
    richer, more diverse context than one single search.

Node contract:
    Reads  : state.plan
    Writes : state.chunks (accumulates, does not overwrite)
"""

from graph.state import AgentState
from retrieval.vector_store import VectorStore
from config import TOP_K_RETRIEVAL
from retrieval.reranker import rerank_chunks_balanced

# The set of tickers we have filings for.
# Used to detect if a sub-task is asking about a specific company.
# If a sub-task contains one of these tickers, we filter Qdrant
# to only return chunks from that company's filings.
KNOWN_TICKERS = {"AAPL", "MSFT", "GOOGL", "NVDA", "META"}

def extract_ticker(text: str) -> str | None:
    """
    Checks if a known ticker is mentioned in a sub-task string.

    This is intentionally simple — we just check if any known
    ticker appears as a word in the text. No NLP needed.

    Example:
        "NVDA risks related to AI competition" → "NVDA"
        "risks related to AI competition"      → None

    Args:
        text: A search sub-task string from the Planner

    Returns:
        Ticker string if found, None if no ticker detected
    """
    # Convert to uppercase so "nvda" and "NVDA" both match
    words = text.upper().split()
    for word in words:
        # Strip punctuation that might be attached to the word
        clean = word.strip(".,;:()")
        if clean in KNOWN_TICKERS:
            return clean
    return None

def deduplicate_chunks(chunks: list[dict]) -> list[dict]:
    """
    Removes duplicate chunks from the accumulated results.

    Why do we need this?
        Multiple sub-tasks might retrieve the same chunk from Qdrant
        — especially if they're asking about related topics.
        Sending duplicate chunks to the Synthesizer wastes tokens
        and can skew the answer toward repeated content.

    How it works:
        We use the first 100 characters of each chunk's text as a
        unique fingerprint. If two chunks share the same fingerprint,
        we keep only the first one.

    Args:
        chunks: List of chunk dicts from Qdrant

    Returns:
        Deduplicated list of chunk dicts
    """
    seen = set()
    unique = []

    for chunk in chunks:
        # Use first 100 chars as fingerprint
        fingerprint = chunk["text"][:100]
        if fingerprint not in seen:
            seen.add(fingerprint)
            unique.append(chunk)

    return unique

def retriever_node(state: AgentState) -> dict:
    """
    LangGraph node function for the Retriever agent.

    Loops through each sub-task in state.plan, searches Qdrant,
    and returns all retrieved chunks combined.

    Args:
        state: Current AgentState (we read state.plan)

    Returns:
        Dict with 'chunks' key containing retrieved chunk dicts
    """
    print(f"\n[Retriever] Executing {len(state.plan)} search sub-task(s)...")

    # Initialise the vector store connection
    store = VectorStore()
    all_chunks = []

    for i, sub_task in enumerate(state.plan, 1):
        # Try to detect a ticker in this sub-task
        ticker = extract_ticker(sub_task)

        if ticker:
            print(f"  [{i}] Searching with filter: {ticker} | Query: {sub_task}")
        else:
            print(f"  [{i}] Searching all companies | Query: {sub_task}")

        # Search Qdrant
        # filter_ticker=None means search across all companies
        results = store.search(
            query=sub_task,
            top_k=TOP_K_RETRIEVAL,
            filter_ticker=ticker,
        )

        print(f"       Retrieved {len(results)} chunks")
        all_chunks.extend(results)

    # Remove duplicates before returning
    unique_chunks = deduplicate_chunks(all_chunks)
    print(f"\n[Retriever] Total chunks after deduplication: {len(unique_chunks)}")

    # Rerank chunks and keep only the top 4
    reranked_chunks = rerank_chunks_balanced(state.query, unique_chunks)
    print(f"[Retriever] Chunks after reranking: {len(reranked_chunks)}")

    # Return chunks — because chunks uses operator.add in AgentState,
    # LangGraph will APPEND these to any existing chunks in state,
    # not overwrite them
    return {"chunks": state.chunks + reranked_chunks}