"""
Utility functions for the RAG system.
"""

from typing import Any, Iterable, List
import re


def pad_embedding_to_1536(embedding: List[float]) -> List[float]:
    """Pad embedding to 1536 dimensions to match database schema."""
    current_dim = len(embedding)
    if current_dim == 1536:
        return embedding
    if current_dim < 1536:
        padding = [0.0] * (1536 - current_dim)
        return embedding + padding
    return embedding[:1536]


def normalize_text(text: str) -> str:
    """Normalize whitespace and control characters before chunking."""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    normalized = re.sub(r"[ \t\f]+", " ", normalized)
    normalized = re.sub(r"\s*\n\s*", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def vector_to_list(vector: Any) -> List[float]:
    """
    Convert embedding model outputs into a plain Python list of floats.
    Supports numpy arrays, torch tensors, and generic iterables.
    """

    if hasattr(vector, "tolist"):
        return list(vector.tolist())  # type: ignore[arg-type]
    if isinstance(vector, list):
        return vector
    if isinstance(vector, Iterable):
        return [float(value) for value in vector]
    raise TypeError(f"Unsupported vector type: {type(vector)!r}")