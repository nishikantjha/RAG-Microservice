# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — Containerizes the TISIX RAG Microservice
#
# WHY Docker?
#   "It works on my machine" is not good enough for production. Docker creates
#   a self-contained environment with the exact Python version, exact package
#   versions, and exact OS settings — identical everywhere it runs.
#
# WHY python:3.11-slim (not python:3.11 full, not alpine)?
#   - python:3.11 full: 1GB+ image, has compilers, debuggers, lots of extras
#   - python:3.11-slim: ~130MB, has just enough to run Python + pip
#   - python:3.11-alpine: Smallest (~50MB) but musl libc breaks many packages
#   → slim is the sweet spot: small enough for fast deployment, compatible enough
#     that sentence-transformers and chromadb install without issues.
#
# NOTE: Ollama runs on the HOST machine (Mac), NOT inside this container.
#   The container calls Ollama via host.docker.internal:11434
#   This is intentional — Ollama needs Metal/GPU access which isn't available
#   inside a Docker container on Mac.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# WHY install gcc/g++ first?
# Some Python packages (chromadb, sentence-transformers) compile C extensions
# during pip install. They need a C compiler. The slim image doesn't include one.
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*
    # rm -rf cleans up apt cache to keep image layer small

# Copy requirements FIRST (before code) — this is a Docker caching trick.
# If you change only your code (not requirements.txt), Docker reuses the
# cached pip install layer and the rebuild is much faster.
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir: don't cache pip downloads in the image (saves ~50MB)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (this layer rebuilds whenever code changes)
COPY . .

# Create directory for ChromaDB persistent storage
# This is where ChromaDB saves its data files
RUN mkdir -p /app/chroma_db

# Tell Docker this container listens on port 8000
# (this is documentation — doesn't actually open the port)
EXPOSE 8000

# Default command to start the server
# --host 0.0.0.0 is required inside Docker (default 127.0.0.1 is unreachable from outside)
# --workers 1 is fine for demo; in production use 2-4 workers
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
