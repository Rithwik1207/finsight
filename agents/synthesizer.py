"""
FinSight — Synthesizer Agent
=============================
The final agent in the Phase 2 pipeline. Takes all retrieved chunks
and generates a coherent, grounded answer to the user's question.

What it does:
    1. Formats all retrieved chunks into a readable context block
    2. Calls OpenAI with the query + context
    3. Extracts unique source references from chunk metadata
    4. Writes the answer and sources into state

Why "grounded"?
    We explicitly instruct the LLM to answer ONLY from the provided
    context — not from its training data. This ensures answers are
    traceable back to real SEC filing passages.

Node contract:
    Reads  : state.query, state.chunks
    Writes : state.answer, state.sources
"""

from openai import OpenAI
from graph.state import AgentState
from config import GROQ_API_KEY, GROQ_MODEL, GROQ_BASE_URL

# Initialise Groq client once at module level (OpenAI-compatible API)
client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)

SYNTHESIZER_SYSTEM_PROMPT = """
You are a financial research analyst specializing in SEC filings analysis.

You will be given:
1. A research question
2. A set of relevant passages retrieved from SEC 10-K filings

Your job is to write a clear, accurate, well-structured answer to the 
research question using ONLY the information in the provided passages.

Rules:
- Base your answer strictly on the provided context
- Do not use your own knowledge or training data
- If the context does not contain enough information, say so clearly
- Be specific — reference company names, numbers, and dates where available
- Write in a professional but clear tone
- Structure your answer in paragraphs, not bullet points
- Keep your answer between 150 and 400 words
"""
def format_context(chunks: list[dict]) -> str:
    """
    Formats retrieved chunks into a readable context block
    to include in the LLM prompt.

    Why format instead of just concatenating text?
        Adding labels like [Source: NVIDIA 10-K 2026-02-25] before
        each passage helps the LLM understand where each piece of
        information comes from — leading to more accurate attribution
        in the final answer.

    Args:
        chunks: List of chunk dicts from state.chunks

    Returns:
        Formatted string with all chunks labelled by source
    """
    formatted = []

    for i, chunk in enumerate(chunks, 1):
        # Build a readable source label from metadata
        source_label = (
            f"{chunk.get('company', 'Unknown')} "
            f"{chunk.get('filing_type', '10-K')} "
            f"{chunk.get('filing_date', '')}"
        )

        formatted.append(
            f"[Passage {i} — Source: {source_label}]\n{chunk['text']}"
        )

    return "\n\n".join(formatted)

def extract_sources(chunks: list[dict]) -> list[str]:
    """
    Extracts unique source references from chunk metadata.

    We deduplicate by source — if 5 chunks came from the same
    filing, we only cite that filing once.

    Example output:
        [
            "Apple Inc. 10-K 2024-11-01",
            "NVIDIA Corporation 10-K 2026-02-25",
        ]

    Args:
        chunks: List of chunk dicts from state.chunks

    Returns:
        Deduplicated list of source reference strings
    """
    seen = set()
    sources = []

    for chunk in chunks:
        source = (
            f"{chunk.get('company', 'Unknown')} "
            f"{chunk.get('filing_type', '10-K')} "
            f"{chunk.get('filing_date', '')}"
        )
        if source not in seen:
            seen.add(source)
            sources.append(source)

    return sources

def synthesizer_node(state: AgentState) -> dict:
    """
    LangGraph node function for the Synthesizer agent.

    Formats retrieved chunks into context, calls OpenAI,
    and returns the final answer with source citations.

    Args:
        state: Current AgentState (we read state.query and state.chunks)

    Returns:
        Dict with 'answer' and 'sources' keys
    """
    print(f"\n[Synthesizer] Generating answer from {len(state.chunks)} chunks...")

    # If no chunks were retrieved, return a graceful fallback
    if not state.chunks:
        print("[Synthesizer] No chunks available — returning fallback answer.")
        return {
            "answer": "I could not find relevant information in the SEC filings to answer this question.",
            "sources": [],
        }

    # If Critic flagged context as insufficient, log it and prepend a warning
    if not state.context_sufficient:
        print(f"[Synthesizer] Warning — Critic flagged weak context: {state.critic_reason}")

    # Format all chunks into a context block for the prompt
    context = format_context(state.chunks)

    # Build the user message combining the question and context
    user_message = f"""Research Question: {state.query}

Relevant passages from SEC 10-K filings:

{context}

Please answer the research question based on the passages above."""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYNTHESIZER_SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            temperature=0.3,  # slightly higher than planner — allows
                              # more natural writing while staying grounded
        )

        answer = response.choices[0].message.content.strip()

        # Prepend a warning if Critic flagged the context
        if not state.context_sufficient:
            answer = (
                f"⚠️ Note: The retrieved context may not fully address this question "
                f"({state.critic_reason})\n\n{answer}"
            )
        sources = extract_sources(state.chunks)

        print(f"[Synthesizer] Answer generated. Sources: {len(sources)}")

        return {
            "answer":  answer,
            "sources": sources,
        }

    except Exception as e:
        print(f"[Synthesizer] Error: {e}")
        return {
            "answer":  f"Error generating answer: {str(e)}",
            "sources": [],
        }