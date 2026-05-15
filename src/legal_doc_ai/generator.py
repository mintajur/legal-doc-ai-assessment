
from __future__ import annotations

import os
from typing import List

from .schemas import DraftResult, RetrievedEvidence, StructuredFields


class GroundedDraftGenerator:
    def __init__(
        self,
        backend: str = "template",
        openai_model: str = "gpt-4o-mini"
    ):
        self.backend = backend
        self.openai_model = openai_model

    def generate(
        self,
        task: str,
        evidence: List[RetrievedEvidence],
        structured_fields: List[StructuredFields]
    ) -> DraftResult:
        if self.backend == "openai" and os.getenv("OPENAI_API_KEY"):
            draft = self._generate_openai(task, evidence, structured_fields)
            backend = "openai"
        else:
            draft = self._generate_template(task, evidence, structured_fields)
            backend = "template"

        return DraftResult(
            task=task,
            draft=draft,
            evidence=evidence,
            structured_fields=structured_fields,
            backend=backend
        )

    def _generate_template(
        self,
        task: str,
        evidence: List[RetrievedEvidence],
        structured_fields: List[StructuredFields]
    ) -> str:
        merged = self._merge_fields(structured_fields)

        tenant = merged["parties"].get("tenant", "Unclear from provided documents")
        landlord = merged["parties"].get("landlord", "Unclear from provided documents")
        property_address = merged.get("property_address") or "Unclear from provided documents"

        monthly_rent = merged["amounts"].get("monthly_rent", "Unclear")
        late_fee = merged["amounts"].get("late_fee", "Unclear")
        balance = merged["amounts"].get("balance", "Unclear")

        notice_date = merged["dates"].get("notice_date", "Unclear")
        lease_start = merged["dates"].get("lease_start", "Unclear")

        issues = sorted(set(merged["issues"]))
        issue_text = ", ".join(issues) if issues else "Unclear"

        evidence_lines = []

        for item in evidence:
            preview = item.text[:350].replace("\n", " ").strip()
            evidence_lines.append(
                f"- [{item.chunk_id}] {item.source_path}, page {item.page_number}: {preview}..."
            )

        evidence_reference_text = ", ".join(
            f"[{item.chunk_id}]" for item in evidence[:3]
        )

        draft = f"""# Notice-Related Case Fact Summary

## Drafting task
{task}

## Short answer
The records indicate possible notice-related issues involving unpaid rent, late payment, and written notice requirements. This summary is limited to the retrieved source evidence and does not make a final legal conclusion.

## Key extracted facts
- Tenant: {tenant}
- Landlord / property manager: {landlord}
- Property: {property_address}
- Notice date: {notice_date}
- Lease start or lease date: {lease_start}
- Monthly rent: {monthly_rent}
- Late fee: {late_fee}
- Stated balance: {balance}
- Main issues found: {issue_text}

## Source-grounded summary
The available records indicate that the tenant and landlord/property manager relationship concerns the property at {property_address}. The documents identify rent-related obligations, including monthly rent of {monthly_rent} and a late-fee provision of {late_fee}. The notice document states that a balance of {balance} was shown for the relevant period. These points are supported by {evidence_reference_text}.

The evidence also indicates that notices under the lease must be provided in writing. Based only on the current documents, the safest conclusion is that the matter appears to involve possible nonpayment, late-payment issues, and written notice requirements. Further legal review is needed before determining the appropriate next step.

## Unsupported or unclear points
- Whether the notice fully complies with local law is not determined from these documents.
- Whether payment was later made is not shown in the retrieved evidence.
- Whether the tenant received the notice is not confirmed by the retrieved evidence.
- No final legal conclusion is made.

## Evidence used
{chr(10).join(evidence_lines)}
"""

        return draft

    def _generate_openai(
        self,
        task: str,
        evidence: List[RetrievedEvidence],
        structured_fields: List[StructuredFields]
    ) -> str:
        from openai import OpenAI

        client = OpenAI()

        evidence_text = "\n\n".join(
            f"[{item.chunk_id}] Source: {item.source_path}, page {item.page_number}\n{item.text}"
            for item in evidence
        )

        structured_text = "\n".join(
            item.model_dump_json(indent=2)
            for item in structured_fields
        )

        prompt = f"""
You are drafting a first-pass internal legal-style notice/case fact summary.

Rules:
1. Use only the provided evidence and structured fields.
2. Do not invent facts.
3. If something is unsupported, write it under "Unsupported or unclear points".
4. Attach evidence IDs like [doc:p1:c0] to material claims.
5. Use cautious wording.
6. Do not make definitive legal conclusions.

Task:
{task}

Structured fields:
{structured_text}

Evidence:
{evidence_text}

Return a clean markdown draft.
"""

        response = client.chat.completions.create(
            model=self.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": "You produce grounded legal-style drafts from retrieved evidence only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1
        )

        return response.choices[0].message.content or ""

    def _merge_fields(
        self,
        structured_fields: List[StructuredFields]
    ) -> dict:
        merged = {
            "parties": {},
            "dates": {},
            "amounts": {},
            "property_address": None,
            "issues": [],
            "uncertainty_notes": []
        }

        for item in structured_fields:
            merged["parties"].update(
                {key: value for key, value in item.parties.items() if value}
            )
            merged["dates"].update(
                {key: value for key, value in item.dates.items() if value}
            )
            merged["amounts"].update(
                {key: value for key, value in item.amounts.items() if value}
            )

            if not merged["property_address"] and item.property_address:
                merged["property_address"] = item.property_address

            merged["issues"].extend(item.issues)
            merged["uncertainty_notes"].extend(item.uncertainty_notes)

        return merged
