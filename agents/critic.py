"""
FinSight — Critic Agent
========================
Evaluates whether the retrieved context is sufficient
to answer the user's query before passing to the Synthesizer.

What it does:
    1. Takes the query and retrieved chunks from state
    2. Calls the LLM to judge context sufficiency
    3. Writes is_sufficient and reason back to state

Node contract:
    Reads  : state.query, state.chunks
    Writes : state.context_sufficient, state.critic_reason
"""

from openai import OpenAI
from graph.state import AgentState
from config import GROQ_API_KEY, GROQ_MODEL, GROQ_BASE_URL
import json

client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)

def build_critic_prompt(query: str, chunks: list[dict]) -> str:
    context_text = ""
    for i, chunk in enumerate(chunks, 1):
        company = chunk.get("company", "Unknown")
        text = chunk.get("text", "")[:400]
        context_text += f"\nChunk {i} [{company}]:\n{text}\n"

    return f"""You are a strict context quality evaluator for a financial research system.

Your job is to decide whether the provided context chunks contain enough information to answer the user's question.

User Question:
{query}

Retrieved Context:
{context_text}

Evaluate the context and respond in JSON format only. No extra text.

{{
    "is_sufficient": true or false,
    "confidence": a float between 0.0 and 1.0,
    "reason": "one sentence explaining your judgment"
}}

Rules:
- is_sufficient is true only if the context directly addresses the question
- is_sufficient is false if the context is only loosely related or completely off-topic
- confidence reflects how certain you are about your judgment
- reason must be one sentence, specific to this query and context"""


def critic_node(state: AgentState) -> dict:
    print(f"\n[Critic] Evaluating context sufficiency...")

    if not state.chunks:
        print("[Critic] No chunks available — marking context as insufficient")
        return {
            "context_sufficient": False,
            "critic_reason": "No chunks were retrieved"
        }

    prompt = build_critic_prompt(state.query, state.chunks)

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()

    try:
        result = json.loads(raw)
        is_sufficient = result.get("is_sufficient", True)
        confidence = result.get("confidence", 1.0)
        reason = result.get("reason", "")

        print(f"[Critic] Sufficient : {is_sufficient}")
        print(f"[Critic] Confidence : {confidence:.2f}")
        print(f"[Critic] Reason     : {reason}")

        return {
            "context_sufficient": is_sufficient,
            "critic_reason": reason
        }

    except json.JSONDecodeError:
        print(f"[Critic] Warning: Could not parse response — defaulting to sufficient")
        return {
            "context_sufficient": True,
            "critic_reason": "Could not parse critic response"
        }