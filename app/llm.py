import json
import logging
from typing import List, Tuple

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ─────────────────────────────────────────────
#  SYSTEM PROMPT — Healthcare RAG Assistant
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are a healthcare information assistant. 
Answer questions using ONLY the provided context.
If the answer is not in the context, say: "I could not find this information in the provided documents."
Never diagnose. Never guess. Be concise and professional."""

HUMAN_PROMPT_TEMPLATE = """Context from healthcare documents:
{context}

Question: {question}

Using the context above, provide a helpful answer in 2-3 sentences:"""

def _build_context_block(retrieved_chunks: List[Tuple[str, dict, float]]) -> str:
    """Format retrieved chunks into a readable context block."""
    if not retrieved_chunks:
        return "No relevant documents found."

    parts = []
    for i, (doc_text, metadata, distance) in enumerate(retrieved_chunks, 1):
        source = metadata.get("source", "unknown")
        parts.append(f"[Source {i}: {source}]\n{doc_text.strip()}")

    return "\n\n---\n\n".join(parts)


class OllamaLLM:
    """
    Client for Ollama-hosted local LLMs (Llama3, Mistral, etc.).
    Uses the /api/chat endpoint with streaming disabled.
    """

    def __init__(self, base_url: str = None, model: str = None):
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_model
        self.timeout = httpx.Timeout(120.0)  # LLMs can be slow

    def health_check(self) -> bool:
        """Check if Ollama is reachable and the model is available."""
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            if resp.status_code != 200:
                return False
            models = [m["name"].split(":")[0] for m in resp.json().get("models", [])]
            available = self.model.split(":")[0] in models
            if not available:
                logger.warning(
                    f"Model '{self.model}' not found in Ollama. "
                    f"Available: {models}. Run: ollama pull {self.model}"
                )
            return available
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False

    def generate(
        self,
        question: str,
        retrieved_chunks: List[Tuple[str, dict, float]],
    ) -> str:
        """
        Generate a grounded answer from retrieved context.

        Args:
            question: The user's question.
            retrieved_chunks: List of (text, metadata, distance) from vector search.

        Returns:
            The LLM-generated answer string.
        """
        context_block = _build_context_block(retrieved_chunks)
        user_message = HUMAN_PROMPT_TEMPLATE.format(
            context=context_block,
            question=question,
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
            "options": {
    "temperature": 0.0,
    "top_p": 0.9,
    "num_predict": 150,
},
        }

        logger.info(f"Sending request to Ollama model='{self.model}'")
        try:
            response = httpx.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            answer = data["message"]["content"].strip()
            logger.info(f"LLM response received ({len(answer)} chars)")
            return answer
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama HTTP error: {e.response.status_code} — {e.response.text}")
            raise RuntimeError(f"LLM request failed: {e.response.status_code}")
        except httpx.RequestError as e:
            logger.error(f"Ollama connection error: {e}")
            raise RuntimeError(
                "Could not connect to Ollama. Ensure it is running: `ollama serve`"
            )


def assess_confidence(retrieved_chunks: List[Tuple[str, dict, float]]) -> str:
    """
    Heuristic confidence rating based on retrieval quality.
    
    Returns: "high" | "medium" | "low" | "none"
    """
    if not retrieved_chunks:
        return "none"

    best_distance = min(d for _, _, d in retrieved_chunks)
    # cosine distance: 0 = identical, 2 = completely opposite
    if best_distance < 0.3:
        return "high"
    elif best_distance < 0.7:
        return "medium"
    elif best_distance < 1.1:
        return "low"
    return "none"
