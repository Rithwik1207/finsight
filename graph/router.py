"""
FinSight — Router
==================
Classifies incoming queries and decides which retrieval
path to take before any search happens.

Routing decisions:
    "retriever"  → use Qdrant SEC filings only
    "web_search" → use Tavily web search only
    "both"       → use both Qdrant and Tavily

The router_node writes the decision to state.route.
The get_route function reads it back for LangGraph's
conditional edge to use.

Node contract:
    Reads  : state.query
    Writes : state.route
"""

from openai import OpenAI
from graph.state import AgentState
from config import GROQ_API_KEY, GROQ_MODEL, GROQ_BASE_URL

client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)

ROUTER_PROMPT = """You are a query router for a financial research system.

You have access to two data sources:
1. SEC 10-K filings database — contains NVIDIA, Microsoft, Apple, Google, Meta filings from 2023-2026
2. Web search — searches the live internet for recent news and current information

Given a user query, decide which source(s) to use.

Respond with exactly one word — nothing else:
- "retriever"  → if the answer is likely in the SEC filings
- "web_search" → if the answer requires recent news or live data not in filings
- "both"       → if the answer needs both filings and recent web data

Rules:
- Questions about risk factors, financials, revenue, strategy → retriever
- Questions about recent news, current events, last week/month → web_search
- Questions comparing filing data with recent developments → both
- When in doubt, use retriever

User query: {query}

Your response (one word only):"""

def router_node(state: AgentState) -> dict:
    """
    LangGraph node for the Router.

    Classifies the query and writes the routing decision
    to state so the conditional edge can read it.

    Node contract:
        Reads  : state.query
        Writes : state.route
    """
    print(f"\n[Router] Classifying query...")

    prompt = ROUTER_PROMPT.format(query=state.query)

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    decision = response.choices[0].message.content.strip().lower()

    # Sanitize — if LLM returns something unexpected, default to retriever
    if decision not in {"retriever", "web_search", "both"}:
        print(f"[Router] Unexpected decision '{decision}' — defaulting to retriever")
        decision = "retriever"

    print(f"[Router] Decision: {decision}")

    return {"route": decision}

def get_route(state: AgentState) -> str:
    """
    Called by LangGraph's conditional edge after router_node runs.
    Returns the routing decision stored in state.
    """
    return state.route