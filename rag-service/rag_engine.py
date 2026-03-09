# rag_engine.py
# ─────────────────────────────────────────────────────────────────────────────
# This is the BRAIN of the service — it handles all AI logic.
#
# ARCHITECTURE:
#   Ingest flow:  Documents → chunk → embed → store in ChromaDB
#   Query flow:   Question → embed → search ChromaDB → get top chunks
#                         → build prompt → send to Ollama LLM → return answer
#
# WHY ChromaDB (not Qdrant, not Pinecone, not FAISS)?
#   - ChromaDB: Pure Python, zero infrastructure, just pip install. Perfect for
#     a trial day demo where you don't want to run extra services.
#   - Qdrant: Better for production (higher performance, Rust-based), but needs
#     a separate running server or Docker container.
#   - Pinecone: Fully managed cloud vector DB, costs money, needs internet.
#   - FAISS: In-memory only (data lost on restart), no built-in metadata search.
#   → ChromaDB wins for this demo because it's persistent AND zero-setup.
#
# WHY all-MiniLM-L6-v2 for embeddings (not OpenAI embeddings, not BGE)?
#   - all-MiniLM-L6-v2: Free, local, 80MB, fast, solid quality (384 dims)
#   - OpenAI text-embedding-3-small: Better quality but costs money per call
#   - BGE-m3: Better multilingual quality but heavier (570MB)
#   → MiniLM wins because it's free, offline, and fast enough for demos.
#
# WHY Ollama (not vLLM, not TGI, not OpenAI API)?
#   - Ollama: One command install, runs on Mac/Linux/Windows, no GPU required
#   - vLLM: Production GPU server, very fast, but NEEDS a GPU
#   - TGI (text-generation-inference): Hugging Face's server, also GPU-focused
#   - OpenAI API: Always available but costs money, not private
#   → Ollama wins for local development and demo on a MacBook.
# ─────────────────────────────────────────────────────────────────────────────

import chromadb
import httpx
import logging
from typing import List, Tuple
from chromadb.utils import embedding_functions

from config import settings
from models import Document

logger = logging.getLogger(__name__)


class RAGEngine:
    """
    Handles the full RAG pipeline:
    1. Ingestion: chunk + embed + store documents
    2. Retrieval: find semantically similar chunks for a query
    3. Generation: use LLM to answer based on retrieved context
    """

    def __init__(self):
        logger.info("Initializing ChromaDB...")

        # PersistentClient saves to disk so data survives restarts
        # Alternative: chromadb.Client() is in-memory only (lost on restart)
        self.client = chromadb.PersistentClient(path=settings.chroma_persist_dir)

        # SentenceTransformer turns text into a vector of numbers (embeddings)
        # Similar texts will have similar vectors — that's how semantic search works
        logger.info("Loading embedding model (all-MiniLM-L6-v2)...")
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        # A "collection" in ChromaDB = a table in SQL = an index in Elasticsearch
        self.collection = self.client.get_or_create_collection(
            name=settings.collection_name,
            embedding_function=self.embedding_fn,
            # cosine distance = good for semantic similarity (text)
            # l2 = euclidean distance (better for image vectors)
            metadata={"hnsw:space": "cosine"}
        )

        logger.info(
            f"ChromaDB ready. Collection '{settings.collection_name}' "
            f"has {self.collection.count()} chunks."
        )

    # ── Text Chunking ──────────────────────────────────────────────────────────

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split a long document into overlapping chunks.

        WHY do we chunk?
          LLMs have a context window limit (e.g., 4096 tokens for small models).
          A full news article might be 2000 tokens. Sending 10 articles at once
          would overflow the context. Chunking lets us send only the RELEVANT
          parts to the LLM, not the entire knowledge base.

        WHY overlapping chunks?
          If a sentence falls exactly at a chunk boundary, it might lose context.
          Overlap ensures the same sentence appears in two adjacent chunks,
          so retrieval can find it either way.
        """
        words = text.split()
        chunks = []

        i = 0
        while i < len(words):
            # Take chunk_size words starting at position i
            chunk_words = words[i: i + settings.chunk_size]
            chunk = " ".join(chunk_words)
            if chunk.strip():
                chunks.append(chunk)
            # Move forward by (chunk_size - chunk_overlap) to create overlap
            i += settings.chunk_size - settings.chunk_overlap

        return chunks

    # ── Ingestion ──────────────────────────────────────────────────────────────

    def ingest(self, documents: List[Document]) -> Tuple[int, int]:
        """
        Process and store documents in ChromaDB.

        Returns: (number of documents, number of chunks created)
        """
        all_ids = []
        all_texts = []
        all_metadatas = []

        for doc in documents:
            chunks = self._chunk_text(doc.content)
            logger.info(f"Document '{doc.id}' → {len(chunks)} chunks")

            # ChromaDB only allows str/int/float/bool in metadata.
            # Convert any list values (e.g. tags) to comma-separated strings.
            safe_metadata = {}
            for k, v in (doc.metadata or {}).items():
                if isinstance(v, list):
                    safe_metadata[k] = ", ".join(str(x) for x in v)
                elif isinstance(v, (str, int, float, bool)):
                    safe_metadata[k] = v
                else:
                    safe_metadata[k] = str(v)

            for i, chunk in enumerate(chunks):
                # Each chunk needs a unique ID — we build it from doc ID + index
                chunk_id = f"{doc.id}__chunk_{i}"
                all_ids.append(chunk_id)
                all_texts.append(chunk)
                all_metadatas.append({
                    **safe_metadata,       # Preserve original metadata (sanitised)
                    "source_id": doc.id,   # Which document this came from
                    "chunk_index": i,      # Which chunk number within the doc
                    "total_chunks": len(chunks),
                })

        if not all_ids:
            return 0, 0

        # upsert = insert OR update if ID already exists
        # This means you can re-ingest updated documents safely
        self.collection.upsert(
            ids=all_ids,
            documents=all_texts,
            metadatas=all_metadatas,
        )

        logger.info(
            f"Ingested {len(documents)} documents as {len(all_ids)} chunks."
        )
        return len(documents), len(all_ids)

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int) -> List[str]:
        """
        Find the most semantically relevant chunks for a given query.

        How it works:
          1. The query is embedded into a vector (same model as during ingestion)
          2. ChromaDB computes cosine similarity between query vector and all stored vectors
          3. Returns the top_k closest chunks
        """
        # Can't retrieve more chunks than what's stored
        n_results = min(top_k, self.collection.count())
        if n_results == 0:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
        )

        # results["documents"] is a list-of-lists (one per query)
        # We sent one query, so we take index [0]
        return results["documents"][0] if results["documents"] else []

    # ── Generation ─────────────────────────────────────────────────────────────

    def generate(self, query: str, context_chunks: List[str]) -> str:
        """
        Send the query + retrieved context to Ollama LLM and get an answer.

        WHY not just ask the LLM directly without context (RAG)?
          Without RAG: LLM answers from its training data (outdated, hallucinated)
          With RAG: LLM answers using YOUR specific documents (accurate, current)
          This is the core value of RAG for enterprise use cases like TISIX.
        """
        # Join chunks with a separator so the LLM can distinguish them
        context_text = "\n\n---\n\n".join(context_chunks)

        # This prompt template is a "system prompt" that tells the LLM its role
        # The [INST] format is specific to Llama models
        prompt = f"""You are a knowledgeable assistant for a publishing and AI consulting company.
Your job is to answer questions accurately using ONLY the provided context.
If the context does not contain enough information to answer, say: "I don't have enough information to answer this based on the available documents."

CONTEXT:
{context_text}

QUESTION: {query}

ANSWER (be concise and factual):"""

        logger.info(f"Calling Ollama model: {settings.ollama_model}")

        # httpx is the modern async HTTP client for Python
        # We use the synchronous version here since FastAPI handles async at the route level
        response = httpx.post(
            f"{settings.ollama_base_url}/api/generate",
            json={
                "model": settings.ollama_model,
                "prompt": prompt,
                "stream": False,          # Get full response at once (not streaming)
                "options": {
                    "temperature": 0.1,   # Low temp = more deterministic/factual answers
                    "num_predict": 512,   # Max tokens in response
                    "top_p": 0.9,
                }
            },
            timeout=120.0,  # 2 minutes max — local LLMs can be slow
        )
        response.raise_for_status()
        return response.json()["response"].strip()

    # ── Full RAG Pipeline ──────────────────────────────────────────────────────

    def query(self, query: str, top_k: int = 3) -> dict:
        """
        The full RAG pipeline in one call:
        Retrieve relevant chunks → Generate answer → Return both
        """
        logger.info(f"RAG query: '{query}' (top_k={top_k})")

        # Step 1: Retrieve
        sources = self.retrieve(query, top_k)

        if not sources:
            return {
                "answer": (
                    "The knowledge base is empty. "
                    "Please ingest documents first via POST /ingest."
                ),
                "sources": [],
            }

        # Step 2: Generate
        answer = self.generate(query, sources)

        return {"answer": answer, "sources": sources}

    # ── Utilities ──────────────────────────────────────────────────────────────

    def count(self) -> int:
        """Return total number of chunks stored in ChromaDB"""
        return self.collection.count()

    def check_ollama(self) -> bool:
        """Check if Ollama server is reachable"""
        try:
            response = httpx.get(
                f"{settings.ollama_base_url}/api/tags",
                timeout=3.0,
            )
            return response.status_code == 200
        except Exception:
            return False
