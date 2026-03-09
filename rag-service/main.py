# main.py
# ─────────────────────────────────────────────────────────────────────────────
# WHY FastAPI (not Flask, not Django)?
#   - FastAPI: Modern, async, auto-generates OpenAPI docs at /docs, built-in
#     Pydantic validation. Industry standard for Python microservices in 2025.
#   - Flask: Synchronous, no auto-validation, more boilerplate. Older choice.
#   - Django: Full-stack web framework — overkill for a microservice/API.
#   → FastAPI wins: it's what companies like Uber, Netflix, Microsoft use for AI APIs.
#
# ENDPOINTS:
#   GET  /health        → Is the service alive? Can it reach Ollama?
#   POST /ingest        → Add documents to the knowledge base
#   POST /query         → Ask a question, get an LLM-powered answer
#   GET  /documents     → How many documents are stored?
#   DELETE /documents   → Clear the knowledge base (for testing)
# ─────────────────────────────────────────────────────────────────────────────

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from models import (
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    HealthResponse,
)
from rag_engine import RAGEngine

# ── Logging ────────────────────────────────────────────────────────────────────
# Always set up logging in a production service — you need to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Lifespan (startup/shutdown) ────────────────────────────────────────────────
# WHY lifespan instead of @app.on_event("startup")?
#   lifespan is the modern FastAPI approach (on_event is deprecated).
#   It uses a context manager so startup and shutdown are in the same function,
#   making it clear what resources are initialized and cleaned up.

rag_engine: RAGEngine = None  # type: ignore


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── STARTUP ──
    global rag_engine
    logger.info("═══ TISIX RAG Service Starting Up ═══")
    logger.info(f"Ollama URL: {settings.ollama_base_url}")
    logger.info(f"LLM Model: {settings.ollama_model}")
    logger.info(f"ChromaDB dir: {settings.chroma_persist_dir}")

    rag_engine = RAGEngine()
    logger.info("═══ Service Ready ═══")

    yield  # ← Everything above runs on startup, everything below on shutdown

    # ── SHUTDOWN ──
    logger.info("═══ TISIX RAG Service Shutting Down ═══")


# ── FastAPI App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="TISIX RAG Microservice",
    description=(
        "A production-ready Retrieval-Augmented Generation service "
        "for publisher and editorial content. Built for TISIX.io.\n\n"
        "**Workflow:**\n"
        "1. `POST /ingest` → Upload your documents\n"
        "2. `POST /query` → Ask questions about them\n"
        "3. `GET /health` → Monitor service status"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS: Allows browser-based frontends to call this API
# In production, replace "*" with your actual frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """
    Health check endpoint.

    Always implement this in production microservices.
    Kubernetes and Docker use this to know if the service is alive.
    Also useful for quickly checking if Ollama is reachable.
    """
    ollama_ok = rag_engine.check_ollama()

    return HealthResponse(
        status="healthy" if ollama_ok else "degraded",
        ollama_connected=ollama_ok,
        documents_in_store=rag_engine.count(),
        model=settings.ollama_model,
    )


@app.post("/ingest", response_model=IngestResponse, tags=["Knowledge Base"])
async def ingest_documents(request: IngestRequest):
    """
    Ingest documents into the vector knowledge base.

    Each document is:
    1. Split into overlapping chunks
    2. Each chunk is converted into an embedding vector
    3. Vectors are stored in ChromaDB for fast semantic search

    Send multiple documents at once for efficiency.
    Re-ingesting the same document ID overwrites it (safe to re-run).
    """
    if not request.documents:
        raise HTTPException(status_code=400, detail="No documents provided")

    try:
        docs_count, chunks_count = rag_engine.ingest(request.documents)
        return IngestResponse(
            message=(
                f"Successfully stored {docs_count} documents as {chunks_count} "
                f"searchable chunks in the knowledge base."
            ),
            documents_ingested=docs_count,
            chunks_created=chunks_count,
        )
    except Exception as e:
        logger.error(f"Ingest failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion error: {str(e)}")


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_knowledge_base(request: QueryRequest):
    """
    Query the knowledge base using RAG.

    Pipeline:
    1. Your question is embedded into a vector
    2. ChromaDB finds the most semantically similar document chunks
    3. Those chunks + your question are sent to the Ollama LLM
    4. The LLM generates a grounded answer based ONLY on retrieved context

    This prevents hallucination because the LLM is anchored to real documents.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        result = rag_engine.query(request.query, request.top_k)
        return QueryResponse(
            query=request.query,
            answer=result["answer"],
            sources=result["sources"],
            model_used=settings.ollama_model,
        )
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Cannot connect to Ollama at {settings.ollama_base_url}. "
                "Is Ollama running? Run: ollama serve"
            ),
        )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="LLM response timed out. The model may be loading. Try again.",
        )
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")


@app.get("/documents", tags=["Knowledge Base"])
async def get_document_stats():
    """
    Get stats about what's in the knowledge base.
    Useful for debugging and monitoring.
    """
    return {
        "total_chunks": rag_engine.count(),
        "collection_name": settings.collection_name,
        "embedding_model": "all-MiniLM-L6-v2",
        "llm_model": settings.ollama_model,
    }


@app.delete("/documents", tags=["Knowledge Base"])
async def clear_knowledge_base():
    """
    Delete all documents from the knowledge base.
    Useful for testing — wipes ChromaDB clean.
    """
    try:
        # Delete the collection and recreate it (fastest way to clear ChromaDB)
        rag_engine.client.delete_collection(settings.collection_name)
        rag_engine.collection = rag_engine.client.get_or_create_collection(
            name=settings.collection_name,
            embedding_function=rag_engine.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        return {"message": "Knowledge base cleared successfully.", "chunks_remaining": 0}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Entry point ────────────────────────────────────────────────────────────────
# This allows running with: python main.py
# In production/Docker, we use: uvicorn main:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
