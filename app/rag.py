import logging
from dataclasses import dataclass, field
from typing import List, Optional

from app.config import get_settings
from app.embeddings import DocumentChunker, EmbeddingModel
from app.llm import OllamaLLM, assess_confidence
from app.vector_store_manager import VectorStore

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class SourceReference:
    document: str
    chunk: str
    chunk_index: int
    similarity_score: float  # 0–1, higher = more relevant


@dataclass
class RAGResponse:
    answer: str
    sources: List[SourceReference]
    confidence: str
    question: str
    retrieved_count: int


class RAGPipeline:
    """
    Full Retrieval-Augmented Generation pipeline.

    Flow:
        1. Receive user question
        2. Embed the question
        3. Retrieve top-K similar chunks from ChromaDB
        4. Pass chunks + question to Ollama LLM
        5. Return grounded answer with source citations
    """

    def __init__(self):
        logger.info("Initializing RAG pipeline components...")
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore(embedding_model=self.embedding_model)
        self.llm = OllamaLLM()
        self.chunker = DocumentChunker()
        logger.info("RAG pipeline ready.")

    def ingest(self, data_dir: str = None, reset: bool = False) -> dict:
        """
        Load documents from data_dir, chunk, embed, and store in vector DB.

        Args:
            data_dir: Path to directory containing .txt/.md documents.
            reset: If True, wipe existing vector store before ingesting.

        Returns:
            Summary dict with counts.
        """
        data_dir = data_dir or settings.data_dir

        if reset:
            logger.info("Reset requested — clearing existing vector store.")
            self.vector_store.reset_collection()

        logger.info(f"Starting ingestion from '{data_dir}'...")
        chunks = self.chunker.load_and_chunk_directory(data_dir)

        unique_sources = list({c.source for c in chunks})
        added_count = self.vector_store.add_chunks(chunks)

        result = {
            "status": "success",
            "documents_ingested": len(unique_sources),
            "chunks_created": added_count,
            "sources": unique_sources,
            "total_in_store": self.vector_store.document_count,
        }
        logger.info(f"Ingestion complete: {result}")
        return result

    def ask(self, question: str) -> RAGResponse:
        """
        Answer a question using RAG.

        Args:
            question: Natural language question from the user.

        Returns:
            RAGResponse with answer, sources, and confidence.
        """
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")

        logger.info(f"Processing question: '{question}'")

        # Step 1: Retrieve relevant chunks
        retrieved = self.vector_store.similarity_search(
            query=question,
            top_k=settings.top_k_results,
            similarity_threshold=settings.similarity_threshold,
        )

        # Step 2: Build source references
        sources = [
            SourceReference(
                document=meta.get("source", "unknown"),
                chunk=doc_text[:300] + ("..." if len(doc_text) > 300 else ""),
                chunk_index=meta.get("chunk_index", 0),
                similarity_score=round(max(0.0, 1.0 - dist / 2.0), 3),
            )
            for doc_text, meta, dist in retrieved
        ]

        # Step 3: Assess confidence before calling LLM
        confidence = assess_confidence(retrieved)

        # Step 4: Generate answer
        if not retrieved:
            answer = "I could not find this information in the provided documents."
        else:
            answer = self.llm.generate(question=question, retrieved_chunks=retrieved)

        return RAGResponse(
            answer=answer,
            sources=sources,
            confidence=confidence,
            question=question,
            retrieved_count=len(retrieved),
        )
