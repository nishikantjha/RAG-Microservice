# models.py
# ─────────────────────────────────────────────────────────────────────────────
# WHY Pydantic models?
#   FastAPI uses Pydantic to automatically validate incoming JSON requests and
#   outgoing responses. If the client sends the wrong type (e.g., a number
#   instead of a string), Pydantic rejects it with a clear error message before
#   your code even runs. This is what separates a real API from a prototype.
# ─────────────────────────────────────────────────────────────────────────────

from pydantic import BaseModel, Field
from typing import Optional, List


# ── Ingest ────────────────────────────────────────────────────────────────────

class Document(BaseModel):
    """A single document to be stored in the knowledge base"""
    id: str = Field(..., description="Unique identifier for this document")
    content: str = Field(..., description="The full text content of the document")
    metadata: Optional[dict] = Field(
        default={},
        description="Extra info like source, date, author, category"
    )


class IngestRequest(BaseModel):
    """Request body for POST /ingest"""
    documents: List[Document] = Field(
        ..., description="List of documents to ingest into the vector store"
    )


class IngestResponse(BaseModel):
    """Response from POST /ingest"""
    message: str
    documents_ingested: int
    chunks_created: int


# ── Query ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Request body for POST /query"""
    query: str = Field(..., description="The question to answer using the knowledge base")
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of relevant chunks to retrieve (1-10)"
    )


class Source(BaseModel):
    """A retrieved document chunk that was used to answer the query"""
    content: str
    metadata: dict


class QueryResponse(BaseModel):
    """Response from POST /query"""
    query: str
    answer: str
    sources: List[str]
    model_used: str


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Response from GET /health"""
    status: str                 # "healthy" or "degraded"
    ollama_connected: bool      # Can we reach the LLM?
    documents_in_store: int     # How many chunks are in ChromaDB
    model: str                  # Which LLM model is configured
