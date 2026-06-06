"""
FinSight — Planner Agent
=========================
The first agent in the pipeline. Receives the user's raw query
and breaks it into a list of specific search sub-tasks.

Why do we need a Planner?
    A vague query gives vague retrieval results. The Planner rewrites
    the query into precise, targeted sub-tasks that the Retriever
    can execute against Qdrant for much better results.

    It also handles multi-company queries by creating one sub-task
    per company — so the Retriever searches each company separately
    and accumulates all relevant chunks.

Example:
    Input : "Compare AI risks across NVIDIA and Microsoft"
    Output: [
        "NVDA risks related to AI competition and data center demand",
        "MSFT risks related to AI competition and cloud services"
    ]

Node contract:
    Reads  : state.query
    Writes : state.plan
"""

import json
from openai import OpenAI
from graph.state import AgentState
from config import GROQ_API_KEY, GROQ_MODEL, GROQ_BASE_URL

# Initialise the Groq client once at module level (OpenAI-compatible API)
# This avoids recreating it on every function call
client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)

# This is the instruction we give to the LLM explaining its job.
# A clear, specific system prompt is critical — vague prompts
# produce vague, unreliable outputs.
PLANNER_SYSTEM_PROMPT = """
You are a financial research planning assistant.

Your job is to take a user's financial research question and break it 
into a list of specific search queries that can be used to retrieve 
relevant information from SEC 10-K filings.

Rules:
- Generate between 1 and 4 search sub-tasks depending on complexity
- Each sub-task should be specific and targeted
- If the question mentions specific companies, create one sub-task per company
- Include the company ticker in each sub-task where relevant
- Keep each sub-task under 15 words

You must respond with ONLY a valid JSON array of strings. Nothing else.
No explanation, no preamble, no markdown.

Example input : "Compare AI competition risks across NVIDIA and Apple"
Example output: ["NVDA risks related to AI competition and market share", 
                 "AAPL risks related to AI competition in devices and services"]
"""

def planner_node(state: AgentState) -> dict:
    """
    LangGraph node function for the Planner agent.

    Node contract: every LangGraph node must:
        1. Accept the current AgentState as input
        2. Return a DICT of only the fields it wants to update
           (not the full state — LangGraph merges the dict into state)

    Args:
        state: Current AgentState (we read state.query)

    Returns:
        Dict with 'plan' key containing list of search sub-tasks
    """
    print(f"\n[Planner] Breaking down query: {state.query}")

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {"role": "user",   "content": state.query},
            ],
            temperature=0.2,  # low temperature = more consistent, focused output
        )

        # Extract the text response
        raw = response.choices[0].message.content.strip()

        # Parse the JSON array
        # If the LLM ignored our instructions and wrapped it in markdown
        # code fences (```json ... ```), strip those out first
        raw = raw.replace("```json", "").replace("```", "").strip()
        plan = json.loads(raw)

        # Safety check — make sure we got a non-empty list of strings
        if not isinstance(plan, list) or not plan:
            raise ValueError("Planner returned empty or invalid plan")

        print(f"[Planner] Generated {len(plan)} sub-tasks:")
        for i, task in enumerate(plan, 1):
            print(f"  {i}. {task}")

        # Return only the fields we're updating
        # LangGraph merges this dict into the existing state
        return {"plan": plan}

    except Exception as e:
        # If anything goes wrong, fall back to using the raw query as the plan
        # This ensures the pipeline never crashes at the Planner stage
        print(f"[Planner] Error: {e}. Falling back to raw query.")
        return {"plan": [state.query]}