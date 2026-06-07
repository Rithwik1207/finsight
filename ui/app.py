"""
FinSight — Streamlit UI
========================
Frontend for the FinSight multi-agent RAG pipeline.
Calls the FastAPI backend and displays results.
"""

import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="FinSight",
    page_icon="📈",
    layout="centered",
)

st.title("📈 FinSight")
st.caption("Multi-agent RAG pipeline for SEC 10-K financial research")

st.divider()

query = st.text_area(
    label="Research Question",
    placeholder="e.g. Compare the AI competition risks mentioned by Microsoft and Google in their 2024 10-K filings.",
    height=100,
)

submitted = st.button("Run Analysis", type="primary")

if submitted:
    if not query.strip():
        st.warning("Please enter a research question.")
    else:
        with st.spinner("Running pipeline..."):
            try:
                response = requests.post(
                    f"{API_URL}/query",
                    json={"query": query},
                    timeout=120,
                )
                response.raise_for_status()
                data = response.json()

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the API. Make sure the FastAPI server is running.")
                st.stop()

            except requests.exceptions.Timeout:
                st.error("Request timed out. The pipeline took too long to respond.")
                st.stop()

            except Exception as e:
                st.error(f"Unexpected error: {e}")
                st.stop()

        # ── Answer ────────────────────────────────────────
        st.subheader("Answer")
        st.write(data["answer"])

        st.divider()

        # ── Sources ───────────────────────────────────────
        st.subheader("Sources")
        for source in data["sources"]:
            st.markdown(f"- {source}")

        st.divider()

        # ── Stats ─────────────────────────────────────────
        st.subheader("Run Stats")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Eval Score",  f"{data['eval_score']:.2f}")
        col2.metric("Latency",     f"{data['latency_seconds']}s")
        col3.metric("Retries",     data["retry_count"])
        col4.metric("Route",       data["route"])