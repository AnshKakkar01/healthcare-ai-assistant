"""
Test suite for the Healthcare AI Assistant.

Run with:
    pytest tests/ -v
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from app.main import app
from app.agent import classify_intent, QueryIntent
from app.embeddings import DocumentChunker


# ─────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ─────────────────────────────────────────────
#  Health Check Tests
# ─────────────────────────────────────────────
class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_required_fields(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "app" in data
        assert "vector_store_chunks" in data
        assert "ollama_model" in data


# ─────────────────────────────────────────────
#  Intent Classification Tests
# ─────────────────────────────────────────────
class TestIntentClassification:
    def test_emergency_intent(self):
        assert classify_intent("I have chest pain and can't breathe") == QueryIntent.EMERGENCY

    def test_appointment_intent(self):
        assert classify_intent("I want to book a cardiology appointment") == QueryIntent.APPOINTMENT_BOOKING

    def test_refill_intent(self):
        assert classify_intent("I need to refill my prescription") == QueryIntent.MEDICATION_REFILL

    def test_general_qa_intent(self):
        assert classify_intent("What is HIPAA?") == QueryIntent.DOCUMENT_QA

    def test_appointment_keyword_slot(self):
        assert classify_intent("Are there available slots for Monday?") == QueryIntent.APPOINTMENT_BOOKING


# ─────────────────────────────────────────────
#  Document Chunker Tests
# ─────────────────────────────────────────────
class TestDocumentChunker:
    def setup_method(self):
        self.chunker = DocumentChunker(chunk_size=200, chunk_overlap=30)

    def test_basic_chunking(self):
        text = "First paragraph content here.\n\nSecond paragraph content here.\n\nThird paragraph content."
        chunks = self.chunker.chunk_text(text, "test_doc.txt")
        assert len(chunks) >= 1
        assert all(len(c.content) <= 300 for c in chunks)  # Allow slight overflow

    def test_chunk_metadata(self):
        text = "A paragraph.\n\nAnother paragraph."
        chunks = self.chunker.chunk_text(text, "sample.txt")
        for chunk in chunks:
            assert chunk.source == "sample.txt"
            assert chunk.total_chunks == len(chunks)

    def test_empty_text(self):
        chunks = self.chunker.chunk_text("", "empty.txt")
        assert len(chunks) == 0

    def test_single_paragraph(self):
        text = "A single short paragraph."
        chunks = self.chunker.chunk_text(text, "short.txt")
        assert len(chunks) == 1
        assert chunks[0].content == text


# ─────────────────────────────────────────────
#  ASK Endpoint Tests (mocked RAG/LLM)
# ─────────────────────────────────────────────
class TestAskEndpoint:
    def test_empty_question_rejected(self, client):
        response = client.post("/ask", json={"question": "ab"})
        # Pydantic min_length=3 should reject this
        assert response.status_code in (422, 400)

    def test_emergency_question_bypasses_rag(self, client):
        with patch("app.main.get_rag") as mock_rag:
            mock_instance = MagicMock()
            mock_rag.return_value = mock_instance
            response = client.post("/ask", json={"question": "I have severe chest pain"})
            data = response.json()
            assert response.status_code == 200
            assert data["intent"] == "emergency"
            assert "911" in data["answer"] or "emergency" in data["answer"].lower()

    def test_ask_requires_non_empty_question(self, client):
        response = client.post("/ask", json={"question": "   "})
        # Should fail validation or return error
        assert response.status_code in (422, 400, 500)


# ─────────────────────────────────────────────
#  Ingest Endpoint Tests
# ─────────────────────────────────────────────
class TestIngestEndpoint:
    def test_ingest_invalid_dir_returns_404(self, client):
        with patch("app.main.get_rag") as mock_rag:
            mock_instance = MagicMock()
            mock_instance.ingest.side_effect = FileNotFoundError("Directory not found")
            mock_rag.return_value = mock_instance
            response = client.post("/ingest", json={"data_dir": "/nonexistent"})
            assert response.status_code == 404
