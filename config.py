import os
from dotenv import load_dotenv

load_dotenv()

# LLM (Groq — OpenAI-compatible API)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Qdrant
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "finsight_filings")

# Embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # lightweight, fast, good quality
EMBEDDING_DIMENSION = 384

# Chunking strategy
CHUNK_SIZE = 800        # characters — tuned for 10-K dense text
CHUNK_OVERLAP = 150     # enough overlap to preserve cross-chunk context

# Retrieval
TOP_K_RETRIEVAL = 10    # fetch top 10 from Qdrant before reranking
TOP_K_RERANKED = 4      # keep top 4 after reranking to pass to LLM

# Evaluation
EVAL_SCORE_THRESHOLD = 0.7  # retry loop triggers below this score

# LangSmith Observability
import os
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "finsight")

# This activates tracing — must be set before any LangChain imports
os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY or ""
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

# Tavily
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

#openai
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
