"""
FinSight — Evaluator Agent (LLM Judge)
========================================
Scores the final answer on quality from 0.0 to 1.0.
If score < EVAL_SCORE_THRESHOLD, the pipeline retries.

Node contract:
    Reads  : state.query, state.answer, state.chunks
    Writes : state.eval_score
"""

from openai import OpenAI
from graph.state import AgentState
from config import GROQ_API_KEY, GROQ_MODEL, GROQ_BASE_URL, EVAL_SCORE_THRESHOLD
import json

client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)

EVALUATOR_PROMPT = """You are a strict answer quality judge for a financial research system.

You will be given:
1. The original research question
2. The generated answer

Score the answer from 0.0 to 1.0 based on these criteria:
- Relevance: Does the answer actually address the question asked?
- Completeness: Does it cover the key aspects of the question?
- Specificity: Does it include specific facts, numbers, or company details?
- Grounding: Does it appear to be based on actual source material, not vague generalities?

Respond in JSON format only. No extra text.

{{
    "score": a float between 0.0 and 1.0,
    "reason": "one sentence explaining the score"
}}

Research Question:
{query}

Generated Answer:
{answer}

Your evaluation:"""

def evaluator_node(state: AgentState) -> dict:
    print(f"\n[Evaluator] Scoring answer quality...")

    if not state.answer:
        print("[Evaluator] No answer to evaluate — scoring 0.0")
        return {"eval_score": 0.0}

    prompt = EVALUATOR_PROMPT.format(
        query=state.query,
        answer=state.answer,
    )

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        raw = response.choices[0].message.content.strip()
        result = json.loads(raw)

        score = float(result.get("score", 0.0))
        reason = result.get("reason", "")

        score = max(0.0, min(1.0, score))

        print(f"[Evaluator] Score  : {score:.2f}")
        print(f"[Evaluator] Reason : {reason}")
        print(f"[Evaluator] Threshold: {EVAL_SCORE_THRESHOLD} → {'PASS' if score >= EVAL_SCORE_THRESHOLD else 'RETRY'}")

        return {"eval_score": score}

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        print(f"[Evaluator] Warning: Could not parse response ({e}) — defaulting to 0.5")
        return {"eval_score": 0.5}