
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    generation_backend: str = os.getenv("GENERATION_BACKEND", "template")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    random_seed: int = int(os.getenv("RANDOM_SEED", "42"))
    chunk_size_chars: int = int(os.getenv("CHUNK_SIZE_CHARS", "900"))
    chunk_overlap_chars: int = int(os.getenv("CHUNK_OVERLAP_CHARS", "150"))
    top_k: int = int(os.getenv("TOP_K", "6"))
