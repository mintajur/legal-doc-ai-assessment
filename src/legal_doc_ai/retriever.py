
from __future__ import annotations

import re
from typing import List

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from .schemas import Chunk, RetrievedEvidence


def simple_tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9$,.]+", text.lower())


class HybridRetriever:
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.embedding_model_name = embedding_model_name
        self.embedding_model = None
        self.chunks = []
        self.bm25 = None
        self.index = None
        self.embeddings = None

    def fit(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("Cannot fit retriever with no chunks.")

        self.chunks = chunks

        tokenized_corpus = [simple_tokenize(chunk.text) for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        embeddings = self.embedding_model.encode(
            [chunk.text for chunk in chunks],
            normalize_embeddings=True,
            show_progress_bar=False
        )

        self.embeddings = np.asarray(embeddings, dtype="float32")

        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def search(self, query: str, top_k: int = 6) -> List[RetrievedEvidence]:
        if self.bm25 is None or self.index is None or self.embedding_model is None:
            raise RuntimeError("Retriever must be fitted before search.")

        top_k = min(top_k, len(self.chunks))

        query_tokens = simple_tokenize(query)

        bm25_scores = self.bm25.get_scores(query_tokens)
        bm25_order = np.argsort(bm25_scores)[::-1][:top_k * 2]

        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False
        )

        query_embedding = np.asarray(query_embedding, dtype="float32")
        vector_scores, vector_indices = self.index.search(query_embedding, top_k * 2)

        candidates = {}

        for rank, index in enumerate(bm25_order):
            candidates[int(index)] = {
                "score": float(bm25_scores[index]) + 1.0 / (rank + 1),
                "method": "bm25"
            }

        for rank, (index, score) in enumerate(
            zip(vector_indices[0], vector_scores[0])
        ):
            index = int(index)

            existing = candidates.get(
                index,
                {
                    "score": 0.0,
                    "method": "vector"
                }
            )

            existing["score"] = (
                float(existing["score"]) + float(score) + 1.0 / (rank + 1)
            )

            existing["method"] = (
                "hybrid" if existing["method"] != "vector" else "vector"
            )

            candidates[index] = existing

        ranked_candidates = sorted(
            candidates.items(),
            key=lambda item: item[1]["score"],
            reverse=True
        )[:top_k]

        evidence = []

        for rank, (index, metadata) in enumerate(ranked_candidates, start=1):
            chunk = self.chunks[index]

            evidence.append(
                RetrievedEvidence(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    source_path=chunk.source_path,
                    page_number=chunk.page_number,
                    text=chunk.text,
                    score=float(metadata["score"]),
                    rank=rank,
                    retrieval_method=metadata["method"]
                )
            )

        return evidence
