"""
numpy-only text embeddings using character 3-gram feature hashing.

No sentence transformers required; relies entirely on numpy for fast numeric
operations and hashlib for deterministic bucket assignment.
"""

import hashlib
from typing import Union

import numpy as np

EMBED_DIM: int = 128


def embed(text: str, dim: int = EMBED_DIM) -> np.ndarray:
    """Return an L2-normalised float32 vector for *text*.

    Strategy: character tri-grams → feature-hash each gram into a bucket
    in [0, dim) with a ±1 sign derived from the hash value, then accumulate
    and L2-normalise.  The result is stable, collision-resistant, and runs
    entirely in numpy / Python stdlib.
    """
    vec = np.zeros(dim, dtype=np.float32)

    if not text or not text.strip():
        return vec

    text = text.lower().strip()
    # Ensure minimum length for tri-gram extraction
    padded = text if len(text) >= 3 else text + " " * (3 - len(text))
    ngrams = [padded[i : i + 3] for i in range(len(padded) - 2)]
    if not ngrams:
        ngrams = [padded]

    for ng in ngrams:
        digest = hashlib.md5(ng.encode("utf-8", errors="ignore")).digest()
        # Use first 8 bytes for bucket index, next byte for sign
        h_int = int.from_bytes(digest[:8], "little")
        idx = h_int % dim
        sign = 1 if digest[8] & 1 else -1
        vec[idx] += sign

    norm = float(np.linalg.norm(vec))
    if norm > 0.0:
        vec /= norm
    return vec


def cosine_similarity(
    a: Union[np.ndarray, list], b: Union[np.ndarray, list]
) -> float:
    """Cosine similarity in [-1, 1] between two vectors."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def embed_to_list(text: str, dim: int = EMBED_DIM) -> list[float]:
    """Embed *text* and return a plain Python list (JSON-serialisable)."""
    return embed(text, dim).tolist()


def best_match(
    query: Union[str, list, np.ndarray],
    candidates: list[tuple[str, list]],
    threshold: float,
) -> tuple[str, float] | None:
    """Return the (name, score) of the best-scoring candidate above *threshold*,
    or None if no candidate exceeds *threshold*.

    *candidates* is a list of (name, embedding_list) pairs.
    *query* may be a string (will be embedded) or an existing vector.
    """
    if isinstance(query, str):
        q_vec = embed(query)
    else:
        q_vec = np.asarray(query, dtype=np.float32)

    best_name: str | None = None
    best_score: float = -1.0
    for name, emb in candidates:
        score = cosine_similarity(q_vec, emb)
        if score > best_score:
            best_score = score
            best_name = name

    if best_name is not None and best_score >= threshold:
        return best_name, best_score
    return None
