
# Legal Document Understanding, Grounded Drafting, Hallucination Removal, and Edit Learning

This project is a Google Colab-first AI Engineer assessment solution.

It processes messy legal-style documents, extracts text and structured fields, retrieves relevant evidence, generates a grounded notice-related case fact summary, removes unsupported hallucinated claims, and improves future drafts from operator edits.

## Main workflow

1. Document processing

Supports TXT, PDF, and image files. Uses embedded PDF text when available. Falls back to OCR for scanned or noisy documents. Saves extracted text, OCR confidence, warnings, and structured fields.

2. Grounded retrieval

Splits documents into page-aware chunks. Uses hybrid retrieval with BM25 and sentence-transformer embeddings. Saves retrieval traces for inspection.

3. Draft generation

Generates a notice-related case fact summary. Uses retrieved evidence and structured fields. Marks unsupported or unclear points instead of guessing.

4. Hallucination removal

Checks material claims against retrieved evidence. Removes unsupported or high-risk legal claims. Saves hallucination_report.json. Produces draft_guarded.md.

5. Improvement from edits

Simulates an operator edit. Learns reusable preferences such as cautious legal wording. Applies those preferences to future drafts. Runs hallucination removal again after edit learning.

6. Evaluation

Measures retrieval recall against synthetic gold facts. Measures grounding coverage by checking evidence references. Checks unsupported claim control. Checks whether edit learning was applied. Reports hallucination guard activity.

## How to run in Google Colab

Run the notebook cells from top to bottom.

Main command:

from legal_doc_ai.pipeline import run_pipeline

run_pipeline(
    input_dir="data/sample_docs",
    output_dir="outputs/sample_run",
    task="Create a notice-related case fact summary.",
    simulate_edit=True,
    use_hallucination_guard=True
)

## Main output files

outputs/sample_run/extracted_documents.jsonl
outputs/sample_run/structured_fields.json
outputs/sample_run/chunks.jsonl
outputs/sample_run/retrieval_trace.json
outputs/sample_run/draft_raw_generated.md
outputs/sample_run/draft_guarded.md
outputs/sample_run/hallucination_report.json
outputs/sample_run/draft_default.md
outputs/sample_run/draft_after_edit_learning.md
outputs/sample_run/hallucination_report_after_edit_learning.json
outputs/sample_run/edit_memory.json
outputs/sample_run/evaluation_report.json

## Assumptions and tradeoffs

The sample documents are synthetic.

The goal is source-grounded drafting, not legal advice.

The deterministic template generator is used by default for reproducibility.

OCR confidence is preserved because messy scans may be partially unclear.

The hallucination guard uses transparent evidence-overlap scoring rather than another LLM judge.

The edit-learning loop stores reusable drafting preferences instead of doing model fine-tuning.

Hybrid retrieval is used because legal documents need both exact keyword matching and semantic search.

## Why hallucination removal matters

Legal-style drafting is risky if the system invents facts or legal conclusions. The hallucination guard removes unsupported material claims and high-risk legal conclusions unless the retrieved evidence strongly supports them. This keeps the draft grounded, inspectable, and safer for operator review.


## Visual Report

The project includes a polished HTML visual report:

visual_report.html

It summarizes:
- pipeline stages
- retrieval and grounding metrics
- hallucination-removal results
- structured extracted fields
- retrieved evidence
- final draft after edit learning

The same report is also saved at:

outputs/sample_run/visual_report.html
