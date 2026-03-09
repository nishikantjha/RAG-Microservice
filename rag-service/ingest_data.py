#!/usr/bin/env python3
"""
ingest_data.py
──────────────────────────────────────────────────────────────────────────────
Standalone script to load the sample articles from data/articles.json
into the ChromaDB vector store.

WHY a separate ingest script?
  In production, ingestion and querying are separate concerns:
  - Ingestion happens when new content is published (via POST /ingest API)
  - This script is a one-shot loader for bootstrapping the demo
  You would NOT embed this logic inside the web server startup.

RUN IT:
  python ingest_data.py

Or via the API directly (after starting the server):
  curl -X POST http://localhost:8000/ingest -H "Content-Type: application/json" -d @data/articles.json
  (Note: the API expects {"documents": [...]} wrapper, this script handles it directly)
──────────────────────────────────────────────────────────────────────────────
"""

import json
import sys
from pathlib import Path

# Add parent dir to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from rag_engine import RAGEngine
from models import Document
from config import settings


def main():
    print("═══ TISIX RAG — Sample Data Ingestion ═══")
    print(f"ChromaDB directory: {settings.chroma_persist_dir}")
    print(f"Collection: {settings.collection_name}")
    print()

    # Load the sample articles JSON file
    data_file = Path(__file__).parent / "data" / "articles.json"
    if not data_file.exists():
        print(f"ERROR: Data file not found at {data_file}")
        sys.exit(1)

    with open(data_file, "r", encoding="utf-8") as f:
        raw_articles = json.load(f)

    print(f"Loaded {len(raw_articles)} articles from {data_file}")
    print()

    # Initialize the RAG engine (this loads ChromaDB and the embedding model)
    print("Initializing RAG Engine...")
    engine = RAGEngine()
    print(f"Current chunks in store: {engine.count()}")
    print()

    # Convert raw JSON into Document Pydantic objects for validation
    documents = [
        Document(
            id=article["id"],
            content=article["content"],
            metadata=article.get("metadata", {}),
        )
        for article in raw_articles
    ]

    # Ingest all documents
    print("Starting ingestion...")
    docs_count, chunks_count = engine.ingest(documents)

    print()
    print("═══ Ingestion Complete ═══")
    print(f"  Documents ingested : {docs_count}")
    print(f"  Chunks created     : {chunks_count}")
    print(f"  Total in store now : {engine.count()}")
    print()
    print("You can now start the server and query the knowledge base:")
    print("  uvicorn main:app --host 0.0.0.0 --port 8000 --reload")
    print()
    print("Then open: http://localhost:8000/docs")


if __name__ == "__main__":
    main()
