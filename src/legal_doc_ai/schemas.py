
from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class PageRecord(BaseModel):
    doc_id: str
    source_path: str
    page_number: int
    text: str
    extraction_method: str
    ocr_confidence: Optional[float] = None
    warnings: List[str] = Field(default_factory=list)


class StructuredFields(BaseModel):
    doc_id: str
    parties: Dict[str, str] = Field(default_factory=dict)
    dates: Dict[str, str] = Field(default_factory=dict)
    amounts: Dict[str, str] = Field(default_factory=dict)
    property_address: Optional[str] = None
    issues: List[str] = Field(default_factory=list)
    uncertainty_notes: List[str] = Field(default_factory=list)


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    source_path: str
    page_number: int
    text: str
    start_char: int
    end_char: int


class RetrievedEvidence(BaseModel):
    chunk_id: str
    doc_id: str
    source_path: str
    page_number: int
    text: str
    score: float
    rank: int
    retrieval_method: str


class DraftResult(BaseModel):
    task: str
    draft: str
    evidence: List[RetrievedEvidence]
    structured_fields: List[StructuredFields]
    backend: str


class EditMemory(BaseModel):
    cautious_legal_language: bool = True
    preferred_phrases: List[str] = Field(
        default_factory=lambda: ["records indicate", "appears", "requires legal review"]
    )
    banned_phrases: List[str] = Field(
        default_factory=lambda: ["must vacate", "violated the lease"]
    )
    learned_notes: List[str] = Field(default_factory=list)


class EvaluationReport(BaseModel):
    retrieval_recall_at_k: float
    grounding_coverage: float
    unsupported_claim_control: bool
    edit_learning_applied: bool
    notes: List[str] = Field(default_factory=list)
