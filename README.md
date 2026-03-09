# TISIX AI Developer — Trial Day Project

> **RAG Microservice + Fine-Tuning Demo**
> Built for TISIX.io trial day interview — Cologne, Germany

---

## Project Structure

```
Prep1/
├── rag-service/                 ← The main microservice (most important!)
│   ├── main.py                  ← FastAPI app with all endpoints
│   ├── rag_engine.py            ← RAG logic: ingest, retrieve, generate
│   ├── models.py                ← Pydantic request/response schemas
│   ├── config.py                ← All settings (reads from .env)
│   ├── ingest_data.py           ← Script to load sample articles
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── data/
│       └── articles.json        ← 10 TISIX-themed publishing articles
│
└── fine-tuning/                 ← Fine-tuning pipeline (conceptual demo)
    ├── prepare_data.py          ← Format JSONL data for training
    ├── train_qlora.py           ← QLoRA training script
    ├── inference.py             ← Test the fine-tuned model
    ├── requirements-finetune.txt
    └── data/
        └── training_data.jsonl  ← 8 labeled instruction examples
```

---

## Part 1: RAG Microservice

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running
- Docker Desktop (for containerized deployment)

### Step 1: Install Ollama and pull model

```bash
# Install Ollama (Mac)
brew install ollama

# Pull the LLM model (2GB download, one time)
ollama pull llama3.2:3b

# Start the Ollama server (keep this running in a separate terminal)
ollama serve
```

### Step 2: Set up Python environment with uv

```bash
cd rag-service

# Create virtual environment
uv venv .venv

# Activate it
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### Step 3: Ingest sample data

```bash
python ingest_data.py
```

### Step 4: Run the service

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000/docs** — you'll see the full interactive API documentation.

### Step 5: Test it

```bash
# Health check
curl http://localhost:8000/health

# Ask a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG and why is it useful for newsrooms?"}'

# Add your own document
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [{
      "id": "my_doc_1",
      "content": "Your document content here...",
      "metadata": {"source": "my_source"}
    }]
  }'
```

### Step 6: Run with Docker

```bash
# Make sure Ollama is running on your Mac first!
docker-compose up --build

# Service is available at http://localhost:8000
```

---

## Part 2: Fine-Tuning (QLoRA)

> **Note:** This requires a GPU (CUDA) for practical training.
> On Mac, the script runs but will be slow (CPU/MPS, no 4-bit quantization).

### Step 1: Accept Llama license

Go to https://huggingface.co/meta-llama/Llama-3.2-3B and accept the license.

Then login:
```bash
huggingface-cli login
```

### Step 2: Set up environment

```bash
cd fine-tuning
uv venv .venv
source .venv/bin/activate
pip install -r requirements-finetune.txt  # Note: use pip for torch (not uv)
```

### Step 3: Prepare data

```bash
python prepare_data.py
```

### Step 4: Train

```bash
python train_qlora.py
```

### Step 5: Test inference

```bash
python inference.py
```

---

## Architecture Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| API framework | FastAPI | Async, auto-docs, Pydantic validation |
| LLM runtime | Ollama | Free, local, no GPU needed, OpenAI-compatible API |
| Vector DB | ChromaDB | Zero setup, persistent, pure Python |
| Embeddings | all-MiniLM-L6-v2 | Free, local, fast, 80MB |
| Fine-tuning | QLoRA | 4x VRAM savings vs full fine-tune, 0.1% params trained |
| Container | Docker | Environment consistency dev→prod |
| Env management | uv | Fast, modern Python package manager |
