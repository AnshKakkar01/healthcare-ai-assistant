import logging
import os
from dataclasses import dataclass
from typing import List

from sentence_transformers import SentenceTransformer

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class DocumentChunk:
    """Represents a single chunk of a document with metadata."""
    content: str
    source: str          # filename
    chunk_index: int
    total_chunks: int
    char_start: int
    char_end: int

    def to_metadata(self) -> dict:
        return {
            "source": self.source,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "char_start": self.char_start,
            "char_end": self.char_end,
        }


class DocumentChunker:
    """Splits documents into overlapping chunks for embedding."""

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

    def chunk_text(self, text: str, source: str) -> List[DocumentChunk]:
        """
        Split text into overlapping chunks. Uses paragraph-aware splitting:
        prefers to break at paragraph boundaries, falls back to word boundaries.
        """
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: List[DocumentChunk] = []
        current_chunk = ""
        current_start = 0
        char_offset = 0

        for para in paragraphs:
            # If adding the paragraph stays under chunk_size, append it
            candidate = (current_chunk + "\n\n" + para).strip()
            if len(candidate) <= self.chunk_size:
                current_chunk = candidate
            else:
                # Flush current chunk if it has content
                if current_chunk:
                    chunks.append(DocumentChunk(
                        content=current_chunk,
                        source=source,
                        chunk_index=len(chunks),
                        total_chunks=0,  # filled in after
                        char_start=current_start,
                        char_end=current_start + len(current_chunk),
                    ))
                    # Start new chunk with overlap from previous
                    overlap_text = current_chunk[-self.chunk_overlap:] if self.chunk_overlap else ""
                    current_start = current_start + len(current_chunk) - len(overlap_text)
                    current_chunk = (overlap_text + "\n\n" + para).strip()
                else:
                    # Paragraph itself is too long — split by sentences
                    sentences = para.replace(". ", ".|").split("|")
                    for sentence in sentences:
                        candidate = (current_chunk + " " + sentence).strip()
                        if len(candidate) <= self.chunk_size:
                            current_chunk = candidate
                        else:
                            if current_chunk:
                                chunks.append(DocumentChunk(
                                    content=current_chunk,
                                    source=source,
                                    chunk_index=len(chunks),
                                    total_chunks=0,
                                    char_start=current_start,
                                    char_end=current_start + len(current_chunk),
                                ))
                                overlap_text = current_chunk[-self.chunk_overlap:] if self.chunk_overlap else ""
                                current_start += len(current_chunk) - len(overlap_text)
                                current_chunk = (overlap_text + " " + sentence).strip()
                            else:
                                current_chunk = sentence
            char_offset += len(para) + 2

        if current_chunk:
            chunks.append(DocumentChunk(
                content=current_chunk,
                source=source,
                chunk_index=len(chunks),
                total_chunks=0,
                char_start=current_start,
                char_end=current_start + len(current_chunk),
            ))

        # Set correct total_chunks
        total = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total

        logger.info(f"Chunked '{source}' into {total} chunks (size={self.chunk_size}, overlap={self.chunk_overlap})")
        return chunks

    def load_and_chunk_directory(self, data_dir: str) -> List[DocumentChunk]:
        """Load all .txt and .pdf text files from a directory and chunk them."""
        all_chunks: List[DocumentChunk] = []
        supported_extensions = (".txt", ".md")

        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        files = [f for f in os.listdir(data_dir) if f.lower().endswith(supported_extensions)]
        if not files:
            raise ValueError(f"No supported documents found in {data_dir}")

        for filename in sorted(files):
            filepath = os.path.join(data_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()
                chunks = self.chunk_text(text, source=filename)
                all_chunks.extend(chunks)
                logger.info(f"Loaded '{filename}' → {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"Failed to load '{filename}': {e}")

        logger.info(f"Total chunks loaded from directory: {len(all_chunks)}")
        return all_chunks


class EmbeddingModel:
    """Wraps SentenceTransformer for generating embeddings."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.embedding_model
        logger.info(f"Loading embedding model: {self.model_name}")
        self._model = SentenceTransformer(self.model_name)
        self.dimension = self._model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model loaded. Dimension: {self.dimension}")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        if not texts:
            return []
        vectors = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return vectors.tolist()

    def embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text string."""
        return self.embed([text])[0]
