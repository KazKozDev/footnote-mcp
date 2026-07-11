"""
Internal runtime settings and shared optional dependency state.

Implementation code lives in focused modules: search, fetch, extract,
rerank, and pipeline.
"""

from __future__ import annotations

# Optional dependencies
try:
    import trafilatura  # noqa: F401

    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False

try:
    from sentence_transformers import CrossEncoder, SentenceTransformer

    HAS_EMBEDDINGS = True
except ImportError:
    SentenceTransformer = None
    CrossEncoder = None
    HAS_EMBEDDINGS = False


# Configurable runtime knobs
NUM_PER_ENGINE = 15
TOP_N_FETCH = 15
MAX_CONTENT_CHARS = 12000
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
TOP_CHUNKS_PER_PAGE = 3
TOTAL_CONTEXT_CHUNKS = 15
FETCH_WORKERS = 10
FETCH_TIMEOUT = 12

IMPERSONATE = [
    "chrome110",
    "chrome116",
    "chrome120",
    "chrome123",
    "chrome124",
    "safari15_5",
    "safari17_0",
]

__all__ = [
    "HAS_TRAFILATURA",
    "HAS_EMBEDDINGS",
    "NUM_PER_ENGINE",
    "TOP_N_FETCH",
    "MAX_CONTENT_CHARS",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "TOP_CHUNKS_PER_PAGE",
    "TOTAL_CONTEXT_CHUNKS",
    "FETCH_WORKERS",
    "FETCH_TIMEOUT",
    "IMPERSONATE",
]
