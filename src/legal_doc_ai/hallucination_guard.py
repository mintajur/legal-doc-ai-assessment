
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Dict, Any

from .schemas import RetrievedEvidence, StructuredFields


@dataclass
class ClaimCheck:
    claim: str
    support_score: float
    supported: bool
    best_evidence_id: str | None
    reason: str


class HallucinationGuard:
    """
    A lightweight evidence-grounding guard.

    Purpose:
    - Detect unsupported claims in the generated draft.
    - Remove or quarantine hallucinated sentences.
    - Save an inspectable hallucination report.
    - Keep only claims that are supported by retrieved evidence.

    This is intentionally transparent and reproducible.
    It does not rely on another LLM to judge hallucination.
    """

    def __init__(
        self,
        min_support_score: float = 0.22,
        min_claim_tokens: int = 5
    ):
        self.min_support_score = min_support_score
        self.min_claim_tokens = min_claim_tokens

        self.high_risk_legal_phrases = [
            "must vacate",
            "violated the lease",
            "clearly liable",
            "legally required",
            "fully compliant",
            "valid notice",
            "invalid notice",
            "breached the lease",
            "eviction is proper",
            "eviction is valid",
            "tenant is liable",
            "landlord is liable"
        ]

        self.safe_uncertainty_phrases = [
            "unclear",
            "not confirmed",
            "not determined",
            "not shown",
            "requires legal review",
            "possible",
            "appears",
            "records indicate",
            "based only on"
        ]

    def guard_draft(
        self,
        draft: str,
        evidence: List[RetrievedEvidence],
        structured_fields: List[StructuredFields]
    ) -> tuple[str, Dict[str, Any]]:
        evidence_items = self._prepare_evidence(evidence, structured_fields)

        sections = self._split_markdown_sections(draft)

        guarded_sections = []
        removed_claims = []
        checked_claims = []

        for heading, body in sections:
            if self._is_exempt_section(heading):
                guarded_sections.append((heading, body))
                continue

            guarded_body, section_removed, section_checked = self._guard_section(
                body=body,
                evidence_items=evidence_items
            )

            guarded_sections.append((heading, guarded_body))
            removed_claims.extend(section_removed)
            checked_claims.extend(section_checked)

        guarded_draft = self._rebuild_sections(guarded_sections)

        if removed_claims:
            guarded_draft += "\n\n## Hallucination guard notes\n"
            guarded_draft += (
                "The following unsupported or high-risk claims were removed "
                "or quarantined because they were not sufficiently supported "
                "by the retrieved evidence:\n"
            )

            for item in removed_claims:
                guarded_draft += f"- {item['claim']}\n"

        report = {
            "guard_enabled": True,
            "min_support_score": self.min_support_score,
            "total_claims_checked": len(checked_claims),
            "supported_claims": sum(1 for item in checked_claims if item["supported"]),
            "removed_or_quarantined_claims": len(removed_claims),
            "removed_claims": removed_claims,
            "claim_checks": checked_claims
        }

        return guarded_draft.strip() + "\n", report

    def _prepare_evidence(
        self,
        evidence: List[RetrievedEvidence],
        structured_fields: List[StructuredFields]
    ) -> list[dict]:
        evidence_items = []

        for item in evidence:
            evidence_items.append(
                {
                    "id": item.chunk_id,
                    "text": item.text,
                    "tokens": self._tokenize(item.text)
                }
            )

        structured_text_parts = []

        for field in structured_fields:
            structured_text_parts.append(field.model_dump_json())

        structured_text = "\n".join(structured_text_parts)

        if structured_text.strip():
            evidence_items.append(
                {
                    "id": "structured_fields",
                    "text": structured_text,
                    "tokens": self._tokenize(structured_text)
                }
            )

        return evidence_items

    def _split_markdown_sections(self, draft: str) -> list[tuple[str, str]]:
        lines = draft.splitlines()

        sections = []
        current_heading = ""
        current_body = []

        for line in lines:
            if line.startswith("#"):
                if current_heading or current_body:
                    sections.append((current_heading, "\n".join(current_body).strip()))
                current_heading = line.strip()
                current_body = []
            else:
                current_body.append(line)

        if current_heading or current_body:
            sections.append((current_heading, "\n".join(current_body).strip()))

        return sections

    def _rebuild_sections(self, sections: list[tuple[str, str]]) -> str:
        parts = []

        for heading, body in sections:
            if heading:
                parts.append(heading)
            if body:
                parts.append(body)

        return "\n\n".join(parts)

    def _is_exempt_section(self, heading: str) -> bool:
        heading_lower = heading.lower()

        exempt_keywords = [
            "unsupported or unclear",
            "evidence used",
            "hallucination guard",
            "drafting task"
        ]

        return any(keyword in heading_lower for keyword in exempt_keywords)

    def _guard_section(
        self,
        body: str,
        evidence_items: list[dict]
    ) -> tuple[str, list[dict], list[dict]]:
        lines = body.splitlines()

        guarded_lines = []
        removed_claims = []
        checked_claims = []

        for line in lines:
            stripped = line.strip()

            if not stripped:
                guarded_lines.append(line)
                continue

            if stripped.startswith("- "):
                claim_text = stripped[2:].strip()
                prefix = "- "
            else:
                claim_text = stripped
                prefix = ""

            if not self._is_material_claim(claim_text):
                guarded_lines.append(line)
                continue

            claim_check = self._check_claim(claim_text, evidence_items)

            checked_claims.append(
                {
                    "claim": claim_check.claim,
                    "support_score": round(claim_check.support_score, 3),
                    "supported": claim_check.supported,
                    "best_evidence_id": claim_check.best_evidence_id,
                    "reason": claim_check.reason
                }
            )

            if claim_check.supported:
                guarded_lines.append(line)
            else:
                removed_claims.append(
                    {
                        "claim": claim_check.claim,
                        "support_score": round(claim_check.support_score, 3),
                        "best_evidence_id": claim_check.best_evidence_id,
                        "reason": claim_check.reason
                    }
                )

        guarded_body = "\n".join(guarded_lines).strip()

        return guarded_body, removed_claims, checked_claims

    def _is_material_claim(self, text: str) -> bool:
        text_lower = text.lower()
        tokens = self._tokenize(text)

        if len(tokens) < self.min_claim_tokens:
            return False

        if any(phrase in text_lower for phrase in self.safe_uncertainty_phrases):
            return True

        has_amount = bool(re.search(r"\$[0-9,]+", text))
        has_date = bool(
            re.search(
                r"\b(?:january|february|march|april|may|june|july|august|"
                r"september|october|november|december)\b",
                text_lower
            )
        )
        has_legal_keyword = any(
            word in text_lower
            for word in [
                "tenant",
                "landlord",
                "notice",
                "rent",
                "lease",
                "fee",
                "balance",
                "property",
                "payment",
                "unpaid",
                "received",
                "complies",
                "legal",
                "vacate",
                "liable"
            ]
        )

        return has_amount or has_date or has_legal_keyword

    def _check_claim(
        self,
        claim: str,
        evidence_items: list[dict]
    ) -> ClaimCheck:
        claim_lower = claim.lower()

        high_risk = any(
            phrase in claim_lower
            for phrase in self.high_risk_legal_phrases
        )

        best_score = 0.0
        best_evidence_id = None

        claim_tokens = self._tokenize(claim)

        for evidence in evidence_items:
            score = self._support_score(claim_tokens, evidence["tokens"])

            if score > best_score:
                best_score = score
                best_evidence_id = evidence["id"]

        if high_risk and best_score < 0.65:
            return ClaimCheck(
                claim=claim,
                support_score=best_score,
                supported=False,
                best_evidence_id=best_evidence_id,
                reason="High-risk legal conclusion without strong evidence support."
            )

        if best_score >= self.min_support_score:
            return ClaimCheck(
                claim=claim,
                support_score=best_score,
                supported=True,
                best_evidence_id=best_evidence_id,
                reason="Claim has enough lexical overlap with retrieved evidence."
            )

        return ClaimCheck(
            claim=claim,
            support_score=best_score,
            supported=False,
            best_evidence_id=best_evidence_id,
            reason="Claim is not sufficiently supported by retrieved evidence."
        )

    def _support_score(
        self,
        claim_tokens: set[str],
        evidence_tokens: set[str]
    ) -> float:
        if not claim_tokens:
            return 0.0

        overlap = claim_tokens.intersection(evidence_tokens)

        important_tokens = {
            token for token in claim_tokens
            if len(token) > 3 or token.startswith("$") or token.isdigit()
        }

        if important_tokens:
            important_overlap = important_tokens.intersection(evidence_tokens)
            return len(important_overlap) / len(important_tokens)

        return len(overlap) / len(claim_tokens)

    def _tokenize(self, text: str) -> set[str]:
        stopwords = {
            "the", "and", "or", "of", "to", "a", "an", "is", "are", "was",
            "were", "in", "on", "for", "with", "by", "from", "that", "this",
            "as", "at", "be", "been", "it", "its", "under", "only", "current"
        }

        tokens = re.findall(r"\$?[a-zA-Z0-9,]+", text.lower())

        cleaned = set()

        for token in tokens:
            token = token.strip(",.")
            if token and token not in stopwords:
                cleaned.add(token)

        return cleaned
