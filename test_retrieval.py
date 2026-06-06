"""
FinSight — Retrieval Test
==========================
Verifies that Qdrant is returning relevant chunks for financial queries.

This is a Phase 1 sanity check — not part of the final pipeline.
Run this after build_index.py to confirm retrieval is working correctly.

What good results look like:
    - Score above 0.4 for the top result
    - Returned chunks are clearly relevant to the query
    - Company and filing date match what you'd expect

Usage:
    python test_retrieval.py
"""

from retrieval.vector_store import VectorStore

# ─────────────────────────────────────────────
# TEST QUERIES
# ─────────────────────────────────────────────
# These are the kinds of questions our agents will ask Qdrant.
# We test a mix of:
#   - General queries (search across all companies)
#   - Company-specific queries (filtered to one ticker)

GENERAL_QUERIES = [
    "What are the main risks related to AI and competition?",
    "How does the company generate revenue?",
    "What risks does the company face from regulation?",
]

FILTERED_QUERIES = [
    # (query, ticker) — should only return chunks from that company
    ("What are NVIDIA's risks related to data center demand?", "NVDA"),
    ("How does Apple describe competition in the smartphone market?", "AAPL"),
    ("What does Meta say about advertising revenue?", "META"),
]


# ─────────────────────────────────────────────
# HELPER: PRINT RESULTS
# ─────────────────────────────────────────────

def print_results(results: list[dict], query: str):
    """Prints search results in a readable format."""
    print(f"\nQuery : {query}")
    print("─" * 55)

    if not results:
        print("  No results returned.")
        return

    for i, r in enumerate(results, 1):
        print(f"\n  Result {i}")
        print(f"  Company : {r['company']} ({r['ticker']})")
        print(f"  Filing  : {r['filing_type']} — {r['filing_date']}")
        print(f"  Score   : {r['score']:.4f}")
        # Print first 200 characters of the chunk so we can judge relevance
        print(f"  Text    : {r['text'][:200]}...")


# ─────────────────────────────────────────────
# MAIN TEST RUNNER
# ─────────────────────────────────────────────

def run_tests():
    print("=" * 55)
    print("  FinSight — Retrieval Test")
    print("=" * 55)

    # Connect to Qdrant
    store = VectorStore()

    # Confirm collection has data before testing
    info = store.get_collection_info()
    print(f"\nCollection : {info['collection']}")
    print(f"Vectors    : {info['total_vectors']}")

    if not info["total_vectors"]:
        print("\nCollection is empty. Run build_index.py first.")
        return

    # ── Test 1: General search across all companies ──
    print("\n\n[ TEST 1 — General Search (no filter) ]")
    print("=" * 55)

    for query in GENERAL_QUERIES:
        # top_k=3 keeps output readable — in the real pipeline we fetch 10
        results = store.search(query, top_k=3)
        print_results(results, query)

    # ── Test 2: Filtered search by ticker ──
    print("\n\n[ TEST 2 — Filtered Search (by company) ]")
    print("=" * 55)

    for query, ticker in FILTERED_QUERIES:
        results = store.search(query, top_k=3, filter_ticker=ticker)
        print_results(results, f"{query} [filter: {ticker}]")

    print("\n" + "=" * 55)
    print("  Retrieval test complete.")
    print("  If scores are above 0.4 and text looks relevant,")
    print("  Phase 1 is complete. Move to Phase 2.")
    print("=" * 55)


if __name__ == "__main__":
    run_tests()