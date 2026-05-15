
from __future__ import annotations

import argparse
from pathlib import Path

from .chunking import TextChunker
from .config import Settings
from .document_processor import DocumentProcessor
from .edit_learner import EditLearner
from .evaluator import Evaluator
from .generator import GroundedDraftGenerator
from .hallucination_guard import HallucinationGuard
from .retriever import HybridRetriever
from .utils import ensure_dir, set_seed, write_json, write_jsonl


def run_pipeline(
    input_dir: str | Path,
    output_dir: str | Path,
    task: str,
    simulate_edit: bool = True,
    use_hallucination_guard: bool = True
) -> dict:
    settings = Settings()
    set_seed(settings.random_seed)

    input_dir = Path(input_dir)
    output_dir = ensure_dir(output_dir)

    processor = DocumentProcessor()

    pages, structured_fields = processor.process_directory(input_dir)

    write_jsonl(output_dir / "extracted_documents.jsonl", pages)

    write_json(
        output_dir / "structured_fields.json",
        [item.model_dump() for item in structured_fields]
    )

    chunker = TextChunker(
        chunk_size_chars=settings.chunk_size_chars,
        overlap_chars=settings.chunk_overlap_chars
    )

    chunks = chunker.chunk_pages(pages)

    write_jsonl(output_dir / "chunks.jsonl", chunks)

    retriever = HybridRetriever(settings.embedding_model)
    retriever.fit(chunks)

    evidence = retriever.search(task, top_k=settings.top_k)

    write_json(
        output_dir / "retrieval_trace.json",
        [item.model_dump() for item in evidence]
    )

    generator = GroundedDraftGenerator(
        backend=settings.generation_backend,
        openai_model=settings.openai_model
    )

    draft_result = generator.generate(
        task=task,
        evidence=evidence,
        structured_fields=structured_fields
    )

    raw_draft = draft_result.draft

    (output_dir / "draft_raw_generated.md").write_text(
        raw_draft,
        encoding="utf-8"
    )

    if use_hallucination_guard:
        guard = HallucinationGuard(
            min_support_score=0.22,
            min_claim_tokens=5
        )

        guarded_draft, hallucination_report = guard.guard_draft(
            draft=raw_draft,
            evidence=evidence,
            structured_fields=structured_fields
        )

        write_json(
            output_dir / "hallucination_report.json",
            hallucination_report
        )

        (output_dir / "draft_guarded.md").write_text(
            guarded_draft,
            encoding="utf-8"
        )

        default_draft = guarded_draft

    else:
        hallucination_report = {
            "guard_enabled": False
        }

        default_draft = raw_draft

    (output_dir / "draft_default.md").write_text(
        default_draft,
        encoding="utf-8"
    )

    edit_learner = EditLearner(output_dir / "edit_memory.json")

    if simulate_edit:
        edited_draft = default_draft.replace(
            "The records indicate possible notice-related issues involving unpaid rent, late payment, and written notice requirements.",
            "The records indicate possible nonpayment, late-payment issues, and written notice requirements."
        )

        edited_draft = edited_draft.replace(
            "before determining the appropriate next step.",
            "before determining the appropriate next step or legal remedy."
        )

        edit_learner.learn_from_edit(default_draft, edited_draft)
    else:
        edit_learner.save()

    improved_draft = edit_learner.apply(default_draft)

    if use_hallucination_guard:
        guard = HallucinationGuard(
            min_support_score=0.22,
            min_claim_tokens=5
        )

        improved_draft, improved_hallucination_report = guard.guard_draft(
            draft=improved_draft,
            evidence=evidence,
            structured_fields=structured_fields
        )

        write_json(
            output_dir / "hallucination_report_after_edit_learning.json",
            improved_hallucination_report
        )

    (output_dir / "draft_after_edit_learning.md").write_text(
        improved_draft,
        encoding="utf-8"
    )

    evaluator = Evaluator()

    report = evaluator.evaluate(
        evidence=evidence,
        draft=default_draft,
        improved_draft=improved_draft
    )

    report_dict = report.model_dump()

    report_dict["hallucination_guard_enabled"] = use_hallucination_guard

    if use_hallucination_guard:
        report_dict["hallucination_claims_checked"] = hallucination_report.get(
            "total_claims_checked",
            0
        )
        report_dict["hallucination_claims_removed"] = hallucination_report.get(
            "removed_or_quarantined_claims",
            0
        )

    write_json(output_dir / "evaluation_report.json", report_dict)

    return {
        "pages_processed": len(pages),
        "chunks_created": len(chunks),
        "evidence_items": len(evidence),
        "output_dir": str(output_dir),
        "hallucination_guard_enabled": use_hallucination_guard,
        "evaluation": report_dict
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run legal document AI assessment pipeline."
    )

    parser.add_argument("--input_dir", type=str, default="data/sample_docs")
    parser.add_argument("--output_dir", type=str, default="outputs/sample_run")
    parser.add_argument(
        "--task",
        type=str,
        default="Create a notice-related case fact summary."
    )
    parser.add_argument("--simulate_edit", action="store_true")
    parser.add_argument("--disable_hallucination_guard", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    summary = run_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        task=args.task,
        simulate_edit=args.simulate_edit,
        use_hallucination_guard=not args.disable_hallucination_guard
    )

    print(summary)


if __name__ == "__main__":
    main()
