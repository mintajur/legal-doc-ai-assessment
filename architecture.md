
# Architecture Overview

## Goal

The goal is to build an inspectable AI workflow that turns messy legal-style documents into a grounded first-pass draft and improves future drafts from operator edits.

## Pipeline

Input documents
↓
DocumentProcessor
- Extracts embedded text from PDFs.
- Falls back to OCR for scanned pages and images.
- Normalizes extracted text.
- Extracts structured fields such as parties, dates, amounts, property address, and issues.

↓
TextChunker
- Creates stable page-aware chunks.
- Preserves document ID and page number for citation.

↓
HybridRetriever
- Uses BM25 for exact keyword matching.
- Uses sentence-transformer embeddings with FAISS for semantic retrieval.
- Merges and ranks evidence.
- Saves retrieval_trace.json so evidence can be inspected.

↓
GroundedDraftGenerator
- Uses structured fields and retrieved evidence.
- Generates a notice-related case fact summary.
- Marks unsupported points clearly.
- Keeps evidence IDs visible in the draft.

↓
HallucinationGuard
- Checks material claims against retrieved evidence.
- Removes unsupported or high-risk legal claims.
- Saves hallucination_report.json.
- Produces a safer guarded draft.

↓
EditLearner
- Captures default draft and edited draft.
- Extracts reusable edit preferences.
- Stores edit memory as JSON.
- Applies cautious language to future drafts.

↓
Evaluator
- Checks retrieval recall.
- Checks grounding coverage.
- Checks unsupported claim control.
- Checks whether edit learning was applied.
- Reports hallucination guard activity.

## Why this design is appropriate

The assessment focuses on document understanding, grounded drafting, and improvement from operator edits. This design makes every stage inspectable by saving intermediate artifacts.

## Key tradeoffs

Rule-based structured extraction is simple and transparent, but not as flexible as schema-constrained LLM extraction.

Template drafting is reproducible, but less natural than LLM-based drafting.

The hallucination guard uses transparent evidence-overlap scoring instead of another LLM judge, which makes it easier to inspect.

Edit learning uses reusable preferences instead of fine-tuning, which is safer and easier to evaluate for a take-home assessment.

Synthetic documents are used because the assessment allows mock data.
