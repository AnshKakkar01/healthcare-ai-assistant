# 🏥 Healthcare AI Assistant

A production-ready, RAG-powered AI assistant that answers healthcare-related questions using local LLMs via Ollama — with zero PHI, full source citations, and an agentic tool-routing layer.

---

## 📐 Architecture Overview

```
User Question
     │
     ▼
┌────────────────────┐
│   FastAPI Layer    │  POST /ask, POST /ingest, GET /health
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│   Agent Router     │  Intent classification (appointment / refill / QA / emergency)
└────────┬───────────┘
         │
   ┌─────┴──────┐
   │            │
   ▼            ▼
Mock Tools    RAG Pipeline
(scheduling,  ┌─────────────────────────────┐
 refill)      │  1. Embed question           │
              │     (sentence-transformers)  │
              │  2. Similarity search        │
              │     (ChromaDB / cosine)      │
              │  3. Build context prompt     │
              │  4. Generate answer          │
              │     (Ollama — Llama3)        │
              │  5. Return + cite sources    │
              └─────────────────────────────┘
```

### Component Summary

| Component | Technology | Role |
|---|---|---|
| API Framework | FastAPI | REST endpoints, validation, middleware |
| LLM | Ollama + Llama3 (local) | Answer generation |
| Embeddings | `all-MiniLM-L6-v2` | Semantic vector creation |
| Vector DB | ChromaDB (persistent) | Similarity search |
| Agent Router | Custom keyword classifier | Intent detection + tool dispatch |
| Mock Tools | Python functions | Appointment & refill simulation |
| Containerization | Docker + Docker Compose | Reproducible deployment |

---

## 🚀 Quick Start (Local — Recommended)

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com/download) installed and running

### Step 1 — Install Ollama and pull the model
```bash
# Install Ollama from https://ollama.com/download, then:
ollama pull llama3
ollama serve   # starts Ollama on http://localhost:11434
```

### Step 2 — Clone and set up the project
```bash
git clone <repo-url>
cd healthcare-ai-assistant

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3 — Configure environment
```bash
cp .env.example .env
# Edit .env if you want to change the model or paths
```

### Step 4 — Start the API
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Step 5 — Ingest documents
```bash
curl -X POST http://localhost:8000/ingest
```

### Step 6 — Ask a question
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Can a patient request a medication refill through telehealth?"}'
```

📖 **Interactive API docs:** http://localhost:8000/docs

---

## 🐳 Docker Deployment

### Option A — Docker Compose (full stack with Ollama)
```bash
docker-compose up --build

# Pull model inside the Ollama container (first time):
docker exec -it healthcare_ollama ollama pull llama3

# Ingest documents:
curl -X POST http://localhost:8000/ingest
```

### Option B — Docker only (if Ollama is running locally)
```bash
docker build -t healthcare-ai-assistant .
docker run -p 8000:8000 \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -v $(pwd)/vector_store:/app/vector_store \
  healthcare-ai-assistant
```

---

## 📡 API Reference

### `GET /health`
Returns service status, vector store stats, and Ollama connectivity.

**Response:**
```json
{
  "status": "healthy",
  "app": "Healthcare AI Assistant",
  "version": "1.0.0",
  "ollama_connected": true,
  "ollama_model": "llama3",
  "vector_store_chunks": 87,
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
}
```

---

### `POST /ingest`
Ingest documents from the data directory into ChromaDB.

**Request (optional body):**
```json
{
  "data_dir": "./data",
  "reset": false
}
```

**Response:**
```json
{
  "status": "success",
  "documents_ingested": 6,
  "chunks_created": 87,
  "sources": [
    "telehealth_guidelines.txt",
    "medication_refill_policy.txt",
    "patient_discharge_instructions.txt",
    "hipaa_privacy_guidelines.txt",
    "appointment_scheduling_policy.txt",
    "insurance_eligibility_faq.txt"
  ],
  "total_in_store": 87,
  "elapsed_seconds": 14.3
}
```

---

### `POST /ask`
Ask a question. Routes through agent, then RAG.

**Request:**
```json
{
  "question": "Can a patient request a medication refill through telehealth?"
}
```

**Response:**
```json
{
  "question": "Can a patient request a medication refill through telehealth?",
  "answer": "Yes, patients can request medication refills through telehealth if the medication is already prescribed and does not require an in-person evaluation. Controlled substances (Schedule II–IV) cannot be prescribed via telehealth. The maximum refill period via telehealth is 90 days.\n\nSource: telehealth_guidelines.txt",
  "sources": [
    {
      "document": "telehealth_guidelines.txt",
      "chunk": "Medication refill requests may be reviewed during telehealth visits. Refills are permitted only for medications that are already prescribed...",
      "chunk_index": 2,
      "similarity_score": 0.91
    }
  ],
  "confidence": "high",
  "intent": "document_qa",
  "tool_result": null,
  "elapsed_seconds": 3.2
}
```

---

## 💬 Sample Questions & Expected Behavior

| Question | Intent | Behavior |
|---|---|---|
| "Can I refill my prescription via telehealth?" | document_qa | RAG answer from telehealth doc |
| "Book me a cardiology appointment for Monday" | appointment_booking | Mock slots + RAG scheduling policy |
| "I need to refill my metformin" | medication_refill | Mock refill status + RAG policy |
| "What are my rights under HIPAA?" | document_qa | RAG answer from HIPAA doc |
| "I have chest pain and can't breathe" | emergency | Immediate 911 message, no RAG |
| "What is the cancellation fee?" | document_qa | RAG answer from scheduling doc |
| "What is the weather today?" | document_qa | "I could not find this information..." |
| "Diagnose my symptoms" | document_qa | Refuses, advises seeing a provider |

---

## 🧠 Technical Design Decisions

### LLM: Ollama + Llama3
- **Why local?** No API keys, no PHI leaves the machine — critical for healthcare compliance.
- Llama3 (8B) balances quality and speed on consumer hardware.
- Alternative: swap `OLLAMA_MODEL=mistral` in `.env` for a lighter model.

### Embedding Model: `all-MiniLM-L6-v2`
- 384-dimension vectors, fast CPU inference (~50ms per chunk).
- Excellent semantic similarity for English healthcare text.
- Runs entirely offline after first download.

### Vector Database: ChromaDB
- Persistent local storage — no server required.
- Cosine similarity search with configurable threshold.
- Zero-config for small datasets; scales to 100K+ chunks.

### Chunking Strategy
- Paragraph-aware splitting: prefers to break at `\n\n` boundaries.
- Falls back to sentence boundaries for very long paragraphs.
- 512-token chunks with 64-token overlap for context continuity.

### Prompting Strategy
The system prompt enforces:
1. **Grounding**: Answer ONLY from provided context.
2. **Refusal**: If context is missing, say so explicitly.
3. **Safety**: Never diagnose; refer to a provider.
4. **Tone**: Professional and clear.
5. **Emergency handling**: Always escalate to 911.

Temperature is set to **0.1** for high factual consistency.

Full prompt is in `app/llm.py` → `SYSTEM_PROMPT` and `HUMAN_PROMPT_TEMPLATE`.

### Agentic Workflow
A lightweight keyword-based router classifies each question into one of four intents:
- **`emergency`** → Immediate response (no RAG), instruct to call 911.
- **`appointment_booking`** → Run `check_available_slots()` mock tool + RAG policy lookup.
- **`medication_refill`** → Run `check_refill_status()` mock tool + RAG policy lookup.
- **`document_qa`** → Straight to RAG.

This demonstrates intent classification + tool dispatch without requiring a heavyweight framework.

---

## 📂 Project Structure

```
healthcare-ai-assistant/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app, endpoints, request/response models
│   ├── rag.py               # RAG pipeline orchestration
│   ├── embeddings.py        # Document chunker + SentenceTransformer wrapper
│   ├── vector_store_manager.py  # ChromaDB persistence + similarity search
│   ├── llm.py               # Ollama client + prompt templates
│   ├── agent.py             # Intent router + mock tools
│   └── config.py            # Pydantic settings from .env
├── data/
│   ├── telehealth_guidelines.txt
│   ├── medication_refill_policy.txt
│   ├── patient_discharge_instructions.txt
│   ├── hipaa_privacy_guidelines.txt
│   ├── appointment_scheduling_policy.txt
│   └── insurance_eligibility_faq.txt
├── vector_store/            # ChromaDB persistent storage (auto-created)
├── tests/
│   └── test_app.py          # Pytest test suite
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── pytest.ini
├── .env.example
└── README.md
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

Test coverage includes:
- Health endpoint validation
- Intent classification (all 4 intents)
- Document chunker (edge cases)
- ASK endpoint — emergency bypass
- Ingest endpoint — invalid directory handling

---

## ⚠️ Limitations & Future Improvements

| Limitation | Future Improvement |
|---|---|
| Keyword-based intent router can misfire | Replace with small classifier or LLM-based router |
| No re-ranking of retrieved chunks | Add cross-encoder re-ranking (e.g., `ms-marco-MiniLM`) |
| Mock appointment/refill tools | Integrate real scheduling APIs (HL7 FHIR, Epic, etc.) |
| Single-turn conversation only | Add session memory for multi-turn dialogue |
| No PDF ingestion | Add `pdfplumber` or `PyMuPDF` for PDF parsing |
| No authentication on API | Add OAuth2 / API key middleware |
| No evaluation metrics | Add RAG evaluation with RAGAS framework |
| Ollama required on host | Bundle a tiny model or add OpenAI fallback |

---

## 🔒 Healthcare Data & Privacy Notes

- **No real PHI is used anywhere in this project.** All documents are synthetic.
- In production, Ollama's local deployment ensures PHI never leaves your infrastructure.
- ChromaDB vectors do not store readable text by default in encrypted deployments.
- HIPAA compliance in production would require: BAAs with vendors, audit logging, encryption at rest, access controls, and breach notification procedures.
- All API inputs should be sanitized and rate-limited in a production deployment.

---

## 📜 License

MIT License — for evaluation/demonstration purposes only. Not for clinical use.
