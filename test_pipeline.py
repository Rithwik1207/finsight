"""
FinSight — Pipeline Test
=========================
Runs the full Phase 2 pipeline end to end for a given query.
Use this to verify all agents are working together correctly.

Usage:
    python test_pipeline.py
"""

from graph.pipeline import run_pipeline


# ── Test queries ───────────────────────────────────────
# Start with a simple single-company query first.
# If that works, try the multi-company query.
TEST_QUERY = input("\nEnter your question: ")


def run_test():
    print("=" * 55)
    print("  FinSight — Pipeline Test")
    print("=" * 55)
    print(f"\nQuery: {TEST_QUERY}\n")

    # Run the full pipeline
    result = run_pipeline(TEST_QUERY)

    print("\nPLAN")
    print("=" * 55)

    for task in result["plan"]:
        print("-", task)

    print("\nFIRST 3 CHUNKS")
    print("=" * 55)

    for i, chunk in enumerate(result["chunks"][:10], start=1):
        print(f"\nChunk {i}")
        print(f"Company : {chunk['company']}")
        print(f"Ticker  : {chunk['ticker']}")
        print(f"Score   : {chunk['score']:.4f}")
        print("-" * 50)
        print(chunk["text"][:500])
        print("\n" + "=" * 70)

    # Print the answer
    print("\n" + "=" * 55)
    print("  ANSWER")
    print("=" * 55)
    print(result["answer"])

    # Print the sources
    print("\n" + "=" * 55)
    print("  SOURCES")
    print("=" * 55)
    for source in result["sources"]:
        print(f"  • {source}")

    print("\n" + "=" * 55)
    print("  STATS")
    print("=" * 55)
    print(f"  Chunks retrieved : {len(result['chunks'])}")
    print(f"  Sources cited    : {len(result['sources'])}")
    print(f"  Plan sub-tasks   : {len(result['plan'])}")
    print(f"  Route taken      : {result['route']}")
    print(f"  Context sufficient: {result['context_sufficient']}")
    print(f"  Eval score       : {result['eval_score']:.2f}")
    print(f"  Retries          : {result['retry_count']}")
    print("=" * 55)

if __name__ == "__main__":
    run_test()