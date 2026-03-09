# config.py
# ─────────────────────────────────────────────────────────────────────────────
# WHY pydantic-settings?
#   Every real production service reads its config from environment variables
#   (not hardcoded values). pydantic-settings lets you define settings as a
#   class and automatically reads them from a .env file or real env vars.
#   This means in Docker you just pass -e OLLAMA_MODEL=llama3.2 and the app
#   picks it up without changing any code. That's production thinking.
# ─────────────────────────────────────────────────────────────────────────────

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Ollama is the local LLM runtime — it runs on the HOST machine (Mac)
    # The Docker container talks to it via this URL
    ollama_base_url: str = "http://localhost:11434"

    # llama3.2:3b = 3 Billion parameter model — fast on MacBook, good quality
    # Alternative: llama3.2:1b (faster, worse quality) or mistral:7b (better, slower)
    ollama_model: str = "llama3.2:3b"

    # ChromaDB will persist data to disk so it survives restarts
    chroma_persist_dir: str = "./chroma_db"

    # Name of the collection inside ChromaDB (like a table in SQL)
    collection_name: str = "publisher_docs"

    # How many words per chunk when splitting documents
    # 500 words ≈ ~700 tokens — fits well in most LLM context windows
    chunk_size: int = 500

    # Overlap prevents losing context at chunk boundaries
    # e.g., a sentence split across two chunks will appear in both
    chunk_overlap: int = 50

    # How many relevant chunks to retrieve before sending to LLM
    top_k: int = 3

    class Config:
        # Load from .env file if it exists (for local dev)
        # In Docker/production you'd set real environment variables instead
        env_file = ".env"
        env_file_encoding = "utf-8"


# Single global instance — imported by other modules
settings = Settings()
