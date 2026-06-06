"""
FinSight — Web Search Agent
============================
Searches the live web using Tavily when SEC filings
don't contain enough information to answer the query.

What it does:
    1. Takes the original query from state
    2. Calls Tavily API with advanced search
    3. Formats results as chunks (same structure as Qdrant chunks)
    4. Returns chunks to be appended to state.chunks

Why same chunk format as Qdrant?
    The Critic and Synthesizer read from state.chunks without
    knowing or caring where chunks came from. Keeping the same
    format means zero changes needed downstream.

Node contract:
    Reads  : state.query
    Writes : state.chunks (accumulates, does not overwrite)
"""

from tavily import TavilyClient
from graph.state import AgentState
from config import TAVILY_API_KEY

client = TavilyClient(api_key=TAVILY_API_KEY)

def search_web(query: str, max_results: int = 5) -> list[dict]:
    """
    Searches the web using Tavily and returns results
    formatted as chunks — same structure as Qdrant chunks.

    Why the same format?
        The Critic and Synthesizer read from state.chunks.
        They don't care where chunks came from.
        Keeping the same format means zero changes downstream.
    """
    try:
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="advanced"
        )

        chunks = []
        for result in response.get("results", []):
            chunks.append({
                "text":         result.get("content", ""),
                "score":        result.get("score", 0.0),
                "ticker":       None,
                "company":      "Web Search",
                "filing_type":  "web",
                "filing_date":  "",
                "chunk_index":  0,
                "source_url":   result.get("url", ""),
                "rerank_score": result.get("score", 0.0),
            })

        return chunks

    except Exception as e:
        print(f"[WebSearch] Error: {e}")
        return []

def web_search_node(state: AgentState) -> dict:
    """
    LangGraph node for the Web Search Agent.

    Uses the original query to search the web.
    Results are added to state.chunks alongside
    any chunks already retrieved from Qdrant.

    Node contract:
        Reads  : state.query
        Writes : state.chunks (accumulates, does not overwrite)
    """
    print(f"\n[WebSearch] Searching web for: {state.query}")

    chunks = search_web(state.query)

    print(f"[WebSearch] Retrieved {len(chunks)} web results")

    return {"chunks": state.chunks + chunks}