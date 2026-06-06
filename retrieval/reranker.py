from sentence_transformers import CrossEncoder
from config import TOP_K_RERANKED

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank_chunks(query: str, chunks: list[dict]) -> list[dict]:
    """Global rerank — best TOP_K_RERANKED chunks across all companies."""
    if not chunks:
        return []

    pairs = [(query, chunk["text"]) for chunk in chunks]
    scores = model.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:TOP_K_RERANKED]


def rerank_chunks_balanced(query: str, chunks: list[dict]) -> list[dict]:
    """
    Balanced rerank — splits chunks by ticker first, reranks each
    group separately, then takes an equal share from each.

    Why?
        For multi-company queries, a global rerank can drop all chunks
        from one company if the other company's language matches the
        query more closely. This ensures every company gets represented.

    How it works:
        1. Group chunks by ticker
        2. Rerank each group independently
        3. Take TOP_K_RERANKED // n_companies from each group
        4. If uneven, fill remaining slots from the highest global scores
    """
    if not chunks:
        return []

    # Group chunks by ticker
    groups: dict[str, list[dict]] = {}
    for chunk in chunks:
        ticker = chunk.get("ticker", "UNKNOWN")
        if ticker not in groups:
            groups[ticker] = []
        groups[ticker].append(chunk)

    n_companies = len(groups)

    # If only one company, fall back to global rerank
    if n_companies == 1:
        return rerank_chunks(query, chunks)

    per_company = max(1, TOP_K_RERANKED // n_companies)
    balanced = []

    for ticker, group_chunks in groups.items():
        pairs = [(query, chunk["text"]) for chunk in group_chunks]
        scores = model.predict(pairs)

        for chunk, score in zip(group_chunks, scores):
            chunk["rerank_score"] = float(score)

        sorted_group = sorted(group_chunks, key=lambda x: x["rerank_score"], reverse=True)
        balanced.extend(sorted_group[:per_company])

    # Sort final balanced list by rerank score descending
    balanced = sorted(balanced, key=lambda x: x["rerank_score"], reverse=True)
    return balanced[:TOP_K_RERANKED]