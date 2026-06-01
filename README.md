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

---

## 📋 Required Deliverables Summary

### 1. LLM Used
- **Model:** `llama3.2:1b` (via Ollama — runs 100% locally, no API key needed)
- **Why:** Smallest Llama3 variant that fits in 8GB RAM. Instruction-following is strong enough for grounded RAG responses. Temperature set to 0.0 for maximum factual consistency.
- **Alternative:** `gemma:2b` for even lower RAM usage, or `llama3:8b` on machines with 16GB+ RAM.

---

### 2. Embedding Model Used
- **Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensions:** 384
- **Why:** Fast CPU inference (~50ms per chunk), excellent semantic similarity for English healthcare text, runs fully offline after first download, only ~90MB in size.
- **Library:** `sentence-transformers` (HuggingFace)

---

### 3. Vector Database Used
- **Database:** ChromaDB (persistent local storage)
- **Similarity metric:** Cosine similarity
- **Why:** Zero-config persistent storage, no separate server needed, excellent Python integration, scales to 100K+ chunks, open source.
- **Storage path:** `./vector_store/` (auto-created on first ingest)
- **Chunk count:** 64 chunks from 6 documents

---

### 4. Prompting Strategy

The prompt is designed in two parts:

**System Prompt** — Sets strict behavioral rules:
```
You are a professional healthcare information assistant.
Answer using ONLY the provided context. Be concise and direct.
Do NOT repeat instructions or source labels in your answer.
If the answer is not in the context, say exactly:
"I could not find this information in the provided documents."
Never provide medical diagnoses. Never guess.
```

**Human Prompt Template** — Injects retrieved context + question:
```
Context from healthcare documents:
{context}

Question: {question}

Using the context above, provide a helpful answer in 2-3 sentences:
```

**Key design decisions:**
- Temperature = 0.0 for factual consistency, no creativity
- `num_predict = 150` to keep answers concise and fast
- Context block includes source labels so LLM knows which document each chunk came from
- Explicit refusal instruction prevents hallucination on unknown topics

---

### 5. Agent / Tool Workflow

A lightweight keyword-based intent router classifies every question into one of 4 intents before hitting RAG:

```
Question
   │
   ▼
Intent Classifier (keyword matching)
   │
   ├── EMERGENCY → Instant 911 message, skip RAG entirely
   │
   ├── APPOINTMENT_BOOKING → check_available_slots(department, date)
   │                         Mock tool returns available slots
   │                         (RAG skipped for booking questions)
   │
   ├── MEDICATION_REFILL → check_refill_status(medication_name)
   │                       Mock tool returns refill eligibility
   │                       + RAG pulls refill policy context
   │
   └── DOCUMENT_QA → Straight to RAG pipeline
```

**Mock Tools implemented:**
- `check_available_slots(department, date)` — Simulates a scheduling system, returns available doctor slots for the requested department
- `check_refill_status(medication_name)` — Simulates pharmacy system, returns refill eligibility and last fill date

**In production** these would connect to real systems like Epic, Cerner, or HL7 FHIR APIs.

---

### 6. Sample Questions and Responses

**Q1: General RAG Query**
```json
Request:  {"question": "Can a patient request a medication refill through telehealth?"}
Response: {
  "answer": "A patient may request a medication refill through telehealth, but it
             is not permitted for controlled substances (Schedule II-IV). Refills
             are limited to medications already prescribed that do not require
             in-person evaluation. The maximum refill period via telehealth is 90 days.",
  "confidence": "high",
  "intent": "medication_refill"
}
```

**Q2: Appointment Booking (Agentic Tool)**
```json
Request:  {"question": "Book a cardiology appointment for Tuesday"}
Response: {
  "answer": "I checked mock appointment availability for Cardiology.
             Available slots: Tuesday June 02 at 10:30 AM with Dr. Singh,
             Tuesday June 02 at 2:00 PM with Dr. Sharma...",
  "intent": "appointment_booking",
  "elapsed_seconds": 0.001
}
```

**Q3: Emergency Detection**
```json
Request:  {"question": "I have chest pain and cant breathe"}
Response: {
  "answer": "⚠️ This appears to be a medical emergency. Please call 911
             immediately or go to the nearest emergency room.",
  "intent": "emergency",
  "confidence": "high"
}
```

**Q4: HIPAA Rights**
```json
Request:  {"question": "What are my rights under HIPAA?"}
Response: {
  "answer": "Under HIPAA, patients have the right to access their medical
             records within 30 days, request amendments to inaccurate information,
             request restrictions on how their PHI is used, and file complaints
             with the Privacy Officer or HHS without fear of retaliation.",
  "confidence": "high",
  "intent": "document_qa"
}
```

**Q5: Unknown Question (Anti-Hallucination)**
```json
Request:  {"question": "What is the weather today?"}
Response: {
  "answer": "I could not find this information in the provided documents.",
  "confidence": "low",
  "intent": "document_qa"
}
```

---

### 7. Dataset / Source Details

All documents are **100% synthetic** — no real PHI or patient data used anywhere.

| File | Content | Chunks |
|---|---|---|
| `telehealth_guidelines.txt` | Telehealth eligibility, refill rules, scheduling, privacy | 9 |
| `medication_refill_policy.txt` | Refill channels, timeframes, controlled substances | 9 |
| `patient_discharge_instructions.txt` | Post-care instructions, emergency signs, wound care | 10 |
| `hipaa_privacy_guidelines.txt` | Patient rights, PHI definition, breach notification | 10 |
| `appointment_scheduling_policy.txt` | Booking, cancellation, departments, walk-ins | 11 |
| `insurance_eligibility_faq.txt` | Copay, deductible, prior auth, billing FAQ | 15 |

**Why synthetic?** The assignment explicitly prohibits real PHI. Synthetic documents give full control over content quality and guarantee HIPAA compliance.

**Public sources referenced** (from assignment's suggested list):
- Document structure modeled after MedlinePlus health topic guidelines
- HIPAA content aligned with HHS.gov official guidelines
- Telehealth policies aligned with CMS telehealth coverage rules

---

### 8. API Examples (curl + PowerShell)

**Health Check:**
```bash
# curl (Linux/Mac)
curl http://localhost:8000/health

# PowerShell (Windows)
Invoke-WebRequest -Uri http://localhost:8000/health -Method GET
```

**Ingest Documents:**
```bash
# curl
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"reset": true}'

# PowerShell
Invoke-WebRequest -Uri http://localhost:8000/ingest -Method POST `
  -ContentType "application/json" -Body '{"reset": true}'
```

**Ask a Question:**
```bash
# curl
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are my rights under HIPAA?"}'

# PowerShell
Invoke-WebRequest -Uri http://localhost:8000/ask -Method POST `
  -ContentType "application/json" `
  -Body '{"question": "What are my rights under HIPAA?"}'
```

**Interactive UI (no curl needed):**
```
http://localhost:8000/docs
```