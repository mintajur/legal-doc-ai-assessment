
from __future__ import annotations

from rapidfuzz import fuzz

from .schemas import EvaluationReport, RetrievedEvidence


class Evaluator:
    def evaluate(
        self,
        evidence: list[RetrievedEvidence],
        draft: str,
        improved_draft: str
    ) -> EvaluationReport:
        gold_facts = [
            "monthly rent",
            "$1,850",
            "late fee",
            "$75",
            "written notice",
            "unpaid rent"
        ]

        evidence_blob = "\n".join(item.text.lower() for item in evidence)

        hits = 0

        for fact in gold_facts:
            if (
                fact.lower() in evidence_blob
                or fuzz.partial_ratio(fact.lower(), evidence_blob) > 85
            ):
                hits += 1

        retrieval_recall_at_k = hits / len(gold_facts)

        evidence_ids = [item.chunk_id for item in evidence]
        referenced_count = sum(
            1 for evidence_id in evidence_ids
            if evidence_id in draft
        )

        grounding_coverage = referenced_count / max(1, len(evidence_ids))

        unsupported_claim_control = (
            "Unsupported or unclear points" in draft
            and "not confirmed" in draft
            and "No final legal conclusion" in draft
        )

        edit_learning_applied = (
            "Operator-edit preferences applied" in improved_draft
            and "Further legal review" in improved_draft
        )

        notes = [
            "Synthetic gold facts are used for a small reviewer-friendly evaluation.",
            "Grounding coverage checks whether retrieved evidence IDs are visible in the draft.",
            "Unsupported claim control checks whether missing facts are explicitly marked unclear.",
            "Edit learning checks whether operator preferences are applied to later drafts."
        ]

        return EvaluationReport(
            retrieval_recall_at_k=round(retrieval_recall_at_k, 3),
            grounding_coverage=round(grounding_coverage, 3),
            unsupported_claim_control=unsupported_claim_control,
            edit_learning_applied=edit_learning_applied,
            notes=notes
        )
