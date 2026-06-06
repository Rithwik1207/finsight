"""
FinSight — Vector Store
========================
Wrapper around the Qdrant client that handles three things:
    1. Creating the Qdrant collection if it doesn't exist
    2. Embedding and storing document chunks (upsert)
    3. Searching for relevant chunks given a query

Key concepts:
    Embedding : Converting text into a list of numbers (vector) that
                represents its meaning. Similar meaning = similar numbers.

    Collection: Like a table in a database. All our chunks live in one
                collection called 'finsight_filings'.

    Upsert    : Insert if new, update if already exists. Safe to run
                multiple times without creating duplicates.

    Payload   : Metadata stored alongside each vector in Qdrant.
                Enables filtering by ticker, date, etc. during search.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from sentence_transformers import SentenceTransformer
from typing import Optional
import uuid
import hashlib

from config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    TOP_K_RETRIEVAL,
)

"""
    Encapsulates all Qdrant operations for FinSight.

    Why a class?
        The Qdrant client and embedding model are expensive to initialize
        (network connection + model loading). Wrapping them in a class
        means we initialize once and reuse across all operations.
    """
class VectorStore:

    def __init__(self):
        # Connect to Qdrant Cloud using URL and API key from .env
        self.client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )

        self.collection_name = QDRANT_COLLECTION_NAME

        # Load the embedding model locally
        # This downloads ~90MB on first run, then cached on your machine
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        # Create the collection if it doesn't already exist
        self._ensure_collection()

        """
        Creates the Qdrant collection if it doesn't exist.

        VectorParams tells Qdrant two things:
            size     : How many numbers per vector (384 for our model)
            distance : How to measure similarity between vectors.
                       COSINE is standard for text — it measures the
                       angle between vectors, not raw magnitude.

        The leading underscore in _ensure_collection is a Python
        convention meaning "internal method — don't call this directly
        from outside the class."
        """

    def _ensure_collection(self):
        existing = [c.name for c in self.client.get_collections().collections]

        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIMENSION,
                    distance=Distance.COSINE,
                ),
            )
            print(f"Created collection : {self.collection_name}")
        else:
            print(f"Using collection   : {self.collection_name}")

        # Create payload index on 'ticker' so filtered search works
        # This is required by Qdrant before you can filter on any field
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="ticker",
            field_schema="keyword",  # "keyword" = exact match filtering
        )

        """
        Converts a list of text strings into a list of vectors.

        We batch all texts together in one call — much faster than
        embedding one at a time.

        Example:
            Input : ["Apple faces competition", "Revenue grew 12%"]
            Output: [[0.23, -0.81, ...], [0.91, 0.12, ...]]
                     ↑ 384 numbers each
        """

    def embed(self, texts: list[str]) -> list[list[float]]:
        
        return self.embedder.encode(
            texts,
            show_progress_bar=False
        ).tolist()

        """
        Embeds and stores a list of chunk dicts into Qdrant.

        What is a PointStruct?
            It's Qdrant's unit of storage. Each point has:
                id     : A unique identifier (we use UUID)
                vector : The embedding (384 numbers)
                payload: The metadata + text stored alongside the vector

        Why upsert in batches of 100?
            Sending 2000 vectors in one HTTP request is risky —
            it can time out. Batching keeps each request small and fast.

        Args:
            chunks: List of dicts from process_filing() in chunker.py
                    Each dict has 'text' and 'metadata' keys.

        Returns:
            Number of chunks successfully stored
        """

    def upsert_documents(self, chunks: list[dict]) -> int:
        
        # Extract just the text strings for batch embedding
        texts = [c["text"] for c in chunks]
        embeddings = self.embed(texts)

        # Build a PointStruct for each chunk
        points = [
            PointStruct(
                id=hashlib.md5(
                    f"{chunks[i]['metadata']['ticker']}"
                    f"{chunks[i]['metadata']['filing_date']}"
                    f"{chunks[i]['metadata']['chunk_index']}"
                    .encode()
                ).hexdigest(),   # unique ID for this chunk
                vector=embeddings[i],   # the 384-number embedding
                payload={
                    # Store the text itself in the payload so we can
                    # retrieve it later — Qdrant stores vectors,
                    # but we need the original text to show the LLM
                    "text": chunks[i]["text"],
                    # Spread all metadata fields into the payload
                    **chunks[i]["metadata"],
                },
            )
            for i in range(len(chunks))
        ]

        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )

        return len(points)

        """
        Finds the most relevant chunks for a given query.

        How it works:
            1. Embed the query into a vector (same model as documents)
            2. Ask Qdrant to find the top_k vectors most similar to it
            3. Optionally filter by ticker before searching
            4. Return the matching chunks with their text and metadata

        What is filter_ticker?
            If provided (e.g. "NVDA"), Qdrant only searches chunks
            where payload.ticker == "NVDA". This is how agents restrict
            search to a specific company's filings.

        Args:
            query        : The question or search string
            top_k        : How many results to return (default 10)
            filter_ticker: Optional ticker to restrict search scope

        Returns:
            List of dicts with text, score, and metadata for each result
        """

    def search(
        self,
        query: str,
        top_k: int = TOP_K_RETRIEVAL,
        filter_ticker: Optional[str] = None,
    ) -> list[dict]:
        
        # Embed the query using the same model used for documents
        # Critical: must use the same model or similarity scores are meaningless
        query_vector = self.embed([query])[0]

        # Build optional metadata filter
        search_filter = None
        if filter_ticker:
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="ticker",
                        match=MatchValue(value=filter_ticker.upper()),
                    )
                ]
            )

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=search_filter,  # None means search everything
            with_payload=True,           # include text + metadata in results
        )

        # Format results into clean dicts
        return [
            {
                "text":         r.payload["text"],
                "score":        r.score,           # cosine similarity 0-1
                "ticker":       r.payload.get("ticker"),
                "company":      r.payload.get("company"),
                "filing_type":  r.payload.get("filing_type"),
                "filing_date":  r.payload.get("filing_date"),
                "chunk_index":  r.payload.get("chunk_index"),
            }
            for r in results
        ]

        """
        Returns basic stats about the collection.
        Used in build_index.py and test_retrieval.py to confirm
        how many vectors are stored after ingestion.
        """

    def get_collection_info(self) -> dict:
    
        info = self.client.get_collection(self.collection_name)
        # points_count is the correct field in newer versions of qdrant-client
        count = getattr(info, "points_count", None) or getattr(info, "vectors_count", None)
        return {
            "collection":    self.collection_name,
            "total_vectors": count,
            "status":        info.status,
        }
    
    """
    Returns basic stats about the collection.
    Used in build_index.py and test_retrieval.py to confirm
    how many vectors are stored after ingestion.
    """

            

