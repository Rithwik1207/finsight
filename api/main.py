"""
FinSight — FastAPI
===================
Serves the FinSight pipeline as an HTTP API.

Endpoints:
    POST /query   — runs the full pipeline and returns the answer
    GET  /health  — confirms the API is running
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import time
import config  # noqa — must import first to activate LangSmith tracing

from graph.pipeline import run_pipeline

app = FastAPI(
    title="FinSight API",
    description="Multi-agent RAG pipeline for SEC 10-K financial research",
    version="1.0.0",
)

class QueryRequest(BaseModel):
    query: str = Field(
        description="The financial research question to answer",
        example="What are Microsoft's main AI competition risks?"
    )

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[str]
    eval_score: float
    retry_count: int
    route: str
    chunks_retrieved: int
    latency_seconds: float

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "FinSight API"}


@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    print(f"\n[API] Received query: {request.query}")

    start = time.time()

    try:
        result = run_pipeline(request.query)
    except Exception as e:
        print(f"[API] Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    latency = round(time.time() - start, 2)

    print(f"[API] Completed in {latency}s | Score: {result['eval_score']:.2f}")

    return QueryResponse(
        query=request.query,
        answer=result["answer"],
        sources=result["sources"],
        eval_score=result["eval_score"],
        retry_count=result["retry_count"],
        route=result["route"],
        chunks_retrieved=len(result["chunks"]),
        latency_seconds=latency,
    )