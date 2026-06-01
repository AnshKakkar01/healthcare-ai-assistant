import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.agent import AgentResponse, QueryIntent, route_and_run
from app.config import get_settings
from app.rag import RAGPipeline, RAGResponse

# ─────────────────────────────────────────────
#  Logging Setup
# ─────────────────────────────────────────────
settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  App State (singleton RAG pipeline)
# ─────────────────────────────────────────────
_rag_pipeline: Optional[RAGPipeline] = None


def get_rag() -> RAGPipeline:
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline


# ─────────────────────────────────────────────
#  Lifespan
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    get_rag()  # Warm up pipeline at startup
    yield
    logger.info("Shutting down...")


# ─────────────────────────────────────────────
#  FastAPI App
# ─────────────────────────────────────────────
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Healthcare AI Assistant using RAG + Ollama LLMs. "
        "Answers questions from a curated set of healthcare policy documents."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
#  Request / Response Models
# ─────────────────────────────────────────────
class IngestRequest(BaseModel):
    data_dir: Optional[str] = Field(None, description="Path to document directory. Defaults to ./data")
    reset: bool = Field(False, description="Wipe existing vector store before ingesting")


class IngestResponse(BaseModel):
    status: str
    documents_ingested: int
    chunks_created: int
    sources: List[str]
    total_in_store: int
    elapsed_seconds: float


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Natural language question")

    class Config:
        json_schema_extra = {
            "example": {"question": "Can a patient request a medication refill through telehealth?"}
        }


class SourceModel(BaseModel):
    document: str
    chunk: str
    chunk_index: int
    similarity_score: float


class ToolResultModel(BaseModel):
    tool_name: Optional[str]
    result: Optional[Dict[str, Any]]


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceModel]
    confidence: str
    intent: str
    tool_result: Optional[ToolResultModel]
    elapsed_seconds: float


# ─────────────────────────────────────────────
#  Middleware — Request Timing
# ─────────────────────────────────────────────
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = round(time.perf_counter() - start, 3)
    response.headers["X-Process-Time"] = str(elapsed)
    return response


# ─────────────────────────────────────────────
#  Endpoints
# ─────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health_check():
    """
    Health check endpoint. Returns service status and vector store stats.
    """
    rag = get_rag()
    ollama_ok = rag.llm.health_check()
    doc_count = rag.vector_store.document_count

    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
        "ollama_connected": ollama_ok,
        "ollama_model": settings.ollama_model,
        "vector_store_chunks": doc_count,
        "embedding_model": settings.embedding_model,
    }


@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_documents(request: IngestRequest = IngestRequest()):
    """
    Ingest documents from the data directory into the vector store.
    - Reads .txt and .md files from `data_dir`
    - Chunks, embeds, and stores them in ChromaDB
    - Use `reset=true` to wipe and re-ingest from scratch
    """
    start = time.perf_counter()
    logger.info(f"POST /ingest — dir='{request.data_dir}', reset={request.reset}")

    try:
        rag = get_rag()
        result = rag.ingest(data_dir=request.data_dir, reset=request.reset)
        elapsed = round(time.perf_counter() - start, 2)
        return IngestResponse(**result, elapsed_seconds=elapsed)

    except FileNotFoundError as e:
        logger.error(f"Ingest failed — directory not found: {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        logger.error(f"Ingest failed — invalid input: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception(f"Ingest failed — unexpected error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/ask", response_model=AskResponse, tags=["Question Answering"])
async def ask_question(request: AskRequest):
    """
    Ask a healthcare-related question.

    The agent:
    1. Classifies the intent (appointment, refill, general QA, emergency)
    2. Runs appropriate tool if needed (mock appointment/refill systems)
    3. Retrieves relevant document chunks from ChromaDB
    4. Generates a grounded answer using the local Ollama LLM
    5. Returns the answer with source citations and confidence
    """
    start = time.perf_counter()
    logger.info(f"POST /ask — question='{request.question[:80]}'")

    try:
        rag = get_rag()

        # Step 1: Route through agent
        agent_response: AgentResponse = route_and_run(request.question)
        logger.info(f"Agent intent='{agent_response.intent}', tool='{agent_response.tool_used}'")

        # Step 2: Handle emergency — skip RAG
        if agent_response.intent == QueryIntent.EMERGENCY:
            elapsed = round(time.perf_counter() - start, 3)
            return AskResponse(
                question=request.question,
                answer=agent_response.tool_result["message"],
                sources=[],
                confidence="high",
                intent=agent_response.intent.value,
                tool_result=None,
                elapsed_seconds=elapsed,
            )

        # Step 3: Run RAG if needed
        rag_result: Optional[RAGResponse] = None
        if agent_response.routed_to_rag and agent_response.intent != QueryIntent.APPOINTMENT_BOOKING:
            if rag.vector_store.document_count == 0:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Vector store is empty. Please call POST /ingest first.",
                )
            rag_result = rag.ask(request.question)

        # Step 4: Compose final answer
        answer_parts = []

        if agent_response.tool_result and agent_response.tool_used:
            tool_summary = _format_tool_result(agent_response.tool_used, agent_response.tool_result)
            answer_parts.append(tool_summary)

        if rag_result:
            answer_parts.append(rag_result.answer)

        final_answer = "\n\n".join(answer_parts) if answer_parts else "I could not find this information in the provided documents."

        elapsed = round(time.perf_counter() - start, 3)
        return AskResponse(
            question=request.question,
            answer=final_answer,
            sources=[
                SourceModel(
                    document=s.document,
                    chunk=s.chunk,
                    chunk_index=s.chunk_index,
                    similarity_score=s.similarity_score,
                )
                for s in (rag_result.sources if rag_result else [])
            ],
            confidence=rag_result.confidence if rag_result else "high",
            intent=agent_response.intent.value,
            tool_result=ToolResultModel(
                tool_name=agent_response.tool_used,
                result=agent_response.tool_result,
            ) if agent_response.tool_used else None,
            elapsed_seconds=elapsed,
        )

    except HTTPException:
        raise
    except RuntimeError as e:
        logger.error(f"/ask failed — LLM error: {e}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        logger.exception(f"/ask failed — unexpected error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


def _format_tool_result(tool_name: str, result: Dict[str, Any]) -> str:
    """Format tool results into a human-readable response fragment."""
    if tool_name == "check_available_slots":
        dept = result.get("department", "your requested department")
        slots = result.get("available_slots", [])
        if not slots:
            return f"I checked the scheduling system for {dept} but found no available slots. Please call {result.get('phone', '1-800-APPT-NOW')} directly."
        slot_lines = "\n".join(
            f"  • {s['date']} at {s['time']} with {s['provider']} ({s['type']})"
            for s in slots[:4]
        )
        return (
            f"I checked mock appointment availability for **{dept}**. "
            f"Here are available slots:\n{slot_lines}\n\n"
            f"Book online at: {result.get('booking_url')} or call {result.get('phone')}."
        )

    elif tool_name == "check_refill_status":
        return (
            f"**Refill Status for {result.get('medication', 'your medication')}:**\n"
            f"  • Status: {result.get('status')}\n"
            f"  • Last filled: {result.get('last_filled')}\n"
            f"  • Refills remaining: {result.get('refills_remaining')}\n"
            f"  • Manage at: {result.get('portal_url')} or call {result.get('pharmacy_phone')}."
        )

    return str(result)


# ─────────────────────────────────────────────
#  Global Exception Handler
# ─────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Please check the logs."},
    )
