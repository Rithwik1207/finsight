"""
FinSight — LangGraph Pipeline
==============================
Defines the agent graph — the assembly line that connects all agents.

What this file does:
    1. Imports all agent functions (each agent is just a Python function)
    2. Creates a LangGraph StateGraph using AgentState as the shared state
    3. Adds each agent as a node
    4. Connects nodes with edges (defines execution order)
    5. Compiles the graph into a runnable pipeline

Routing logic:
    "retriever"  → Qdrant only  → critic → synthesizer
    "web_search" → Tavily only  → critic → synthesizer
    "both"       → Qdrant first → web_search → critic → synthesizer
"""

from langgraph.graph import StateGraph, END
from graph.state import AgentState
from graph.router import router_node, get_route
from agents.planner import planner_node
from agents.retriever import retriever_node
from agents.web_search import web_search_node
from agents.critic import critic_node
from agents.synthesizer import synthesizer_node
from agents.evaluator import evaluator_node
from config import EVAL_SCORE_THRESHOLD

def should_retry(state: AgentState) -> str:
    if state.eval_score >= EVAL_SCORE_THRESHOLD:
        return "end"
    if state.retry_count >= 2:
        print(f"[Pipeline] Max retries reached - forcing END" )
        return "end"
    return "retry"

def prepare_retry(state: AgentState) -> dict:
    new_count = state.retry_count + 1
    print(f"\n[Pipeline] Retry {new_count} - clearing atale slate and re-planning...")
    return{
        "retry_count": new_count,
        "chunks" : [],
        "answer": "",
        "sources":[],
        "eval_score": 0.0,
        "context_sufficient": True,
        "critic_reason": "",
    }

def build_graph():
    """
    Assembles and compiles the LangGraph pipeline.

    Why a function instead of module-level code?
        Wrapping it in a function means the graph is only built
        when you call build_graph() — not every time the file
        is imported. Cleaner and avoids circular import issues.

    Returns:
        A compiled LangGraph app ready to invoke with a query.
    """

    # Create the graph, telling it AgentState is the shared state object
    # Every node will receive AgentState and must return a dict of updates
    graph = StateGraph(AgentState)

    # ── Add nodes ──────────────────────────────────────
    # First argument : the name of the node (used in edges)
    # Second argument: the function to call when this node runs
    graph.add_node("planner",     planner_node)
    graph.add_node("router",    router_node)
    graph.add_node("retriever",   retriever_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("critic",      critic_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("evaluator",   evaluator_node)
    graph.add_node("prepare_retry", prepare_retry)

    # ── Add edges ──────────────────────────────────────
    # This defines the execution order.
    # Read as: "after planner finishes, run retriever next"
    graph.add_edge("planner",     "router")  #fixed edge

    # Conditional edge — Router decides the next node
    graph.add_conditional_edges(
        "router",       # which node triggers the condition
        get_route,      # function that reads state.route and returns a string
        {
            "retriever":  "retriever",   # → go to retriever node
            "web_search": "web_search",  # → go to web_search node
            "both":       "retriever",   # → go to retriever first, then web_search
        }
    )

    # After retriever:
    # - if route is "retriever" → go straight to critic
    # - if route is "both"      → go to web_search first
    graph.add_conditional_edges(
        "retriever",
        get_route,
        {
            "retriever":  "critic",      # retriever only → done retrieving
            "web_search": "critic",      # shouldn't happen, safety fallback
            "both":       "web_search",  # both → chain into web_search next
        }
    )

    # After web_search → always go to critic
    graph.add_edge("web_search", "critic")

    # Fixed edges after critic
    graph.add_edge("critic",    "synthesizer")
    graph.add_edge("synthesizer", "evaluator")

    # Retry loop — the new conditional edge
    graph.add_conditional_edges(
        "evaluator",
        should_retry,
        {
            "end":   END,
            "retry": "prepare_retry",
        }
    )

    graph.add_edge("prepare_retry", "planner")

    # ── Set entry point ────────────────────────────────
    # Tells LangGraph which node to run first
    graph.set_entry_point("planner")

    # ── Compile ────────────────────────────────────────
    # Compiles the graph into a runnable app.
    # After this, you can call app.invoke({"query": "..."})
    app = graph.compile()

    return app

def run_pipeline(query: str) -> dict:
    """
    Runs the full pipeline for a given user query.

    This is the single entry point for everything —
    FastAPI will call this, Streamlit will call this.
    Nothing outside this file needs to know about nodes or edges.

    Args:
        query: The user's financial research question

    Returns:
        The final AgentState with answer, sources, and eval_score filled in
    """

    # Build the graph
    app = build_graph()

    # Create the initial state with just the query
    # All other fields start at their defaults (empty lists, empty strings)
    initial_state = AgentState(query=query)

    # Run the pipeline
    # invoke() runs all nodes in order and returns the final state
    final_state = app.invoke(initial_state)
    return final_state