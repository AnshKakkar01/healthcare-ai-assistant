import logging
import uuid
from typing import List, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings
from app.embeddings import DocumentChunk, EmbeddingModel

logger = logging.getLogger(__name__)
settings = get_settings()


class VectorStore:
    """
    ChromaDB-backed vector store for healthcare document chunks.
    Handles persistence, upsert, and similarity search.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        persist_dir: str = None,
        collection_name: str = None,
    ):
        self.embedding_model = embedding_model
        self.persist_dir = persist_dir or settings.vector_store_path
        self.collection_name = collection_name or settings.collection_name

        self._client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"VectorStore ready. Collection='{self.collection_name}', "
            f"Docs={self._collection.count()}, Path='{self.persist_dir}'"
        )

    @property
    def document_count(self) -> int:
        return self._collection.count()

    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        """
        Embed and upsert a list of DocumentChunks into the collection.
        Returns the number of chunks added.
        """
        if not chunks:
            logger.warning("add_chunks called with empty list.")
            return 0

        texts = [c.content for c in chunks]
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedding_model.embed(texts)

        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [c.to_metadata() for c in chunks]

        # Upsert in batches of 500 to avoid memory issues
        batch_size = 500
        for i in range(0, len(chunks), batch_size):
            batch_slice = slice(i, i + batch_size)
            self._collection.upsert(
                ids=ids[batch_slice],
                embeddings=embeddings[batch_slice],
                documents=texts[batch_slice],
                metadatas=metadatas[batch_slice],
            )
            logger.debug(f"Upserted batch {i // batch_size + 1}")

        logger.info(f"Successfully added {len(chunks)} chunks. Total in store: {self._collection.count()}")
        return len(chunks)

    def similarity_search(
        self,
        query: str,
        top_k: int = None,
        similarity_threshold: float = None,
    ) -> List[Tuple[str, dict, float]]:
        """
        Search the vector store for chunks similar to `query`.

        Returns a list of (document_text, metadata, distance) tuples,
        filtered by similarity_threshold (cosine distance; lower = more similar).
        """
        top_k = top_k or settings.top_k_results
        threshold = similarity_threshold if similarity_threshold is not None else settings.similarity_threshold

        if self._collection.count() == 0:
            logger.warning("Vector store is empty. Run /ingest first.")
            return []

        query_embedding = self.embedding_model.embed_single(query)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        # Filter by threshold (cosine distance: 0 = identical, 2 = opposite)
        filtered = [
            (doc, meta, dist)
            for doc, meta, dist in zip(documents, metadatas, distances)
            if dist <= (2.0 - threshold * 2)  # convert similarity threshold to distance
        ]

        logger.info(
            f"Search for '{query[:60]}...' → {len(filtered)}/{top_k} results "
            f"(threshold={threshold})"
        )
        return filtered

    def reset_collection(self) -> None:
        """Delete and recreate the collection — used for re-ingestion."""
        logger.warning(f"Resetting collection '{self.collection_name}'...")
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Collection reset complete.")
