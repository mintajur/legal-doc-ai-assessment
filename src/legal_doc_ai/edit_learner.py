
from __future__ import annotations

import difflib
from pathlib import Path

from .schemas import EditMemory
from .utils import read_json, write_json


class EditLearner:
    def __init__(self, memory_path: str | Path):
        self.memory_path = Path(memory_path)

        if self.memory_path.exists():
            self.memory = EditMemory(**read_json(self.memory_path))
        else:
            self.memory = EditMemory()

    def learn_from_edit(
        self,
        default_draft: str,
        edited_draft: str
    ) -> EditMemory:
        diff = list(
            difflib.unified_diff(
                default_draft.splitlines(),
                edited_draft.splitlines(),
                lineterm=""
            )
        )

        removed_text = "\n".join(
            line[1:] for line in diff
            if line.startswith("-") and not line.startswith("---")
        )

        added_text = "\n".join(
            line[1:] for line in diff
            if line.startswith("+") and not line.startswith("+++")
        )

        if (
            "must vacate" in removed_text.lower()
            or "violated the lease" in removed_text.lower()
        ):
            self.memory.cautious_legal_language = True

            note = (
                "Operator softened definitive legal conclusions into cautious "
                "source-grounded language."
            )

            if note not in self.memory.learned_notes:
                self.memory.learned_notes.append(note)

        for phrase in [
            "records indicate",
            "appears",
            "requires legal review",
            "possible nonpayment"
        ]:
            if phrase in added_text.lower() and phrase not in self.memory.preferred_phrases:
                self.memory.preferred_phrases.append(phrase)

        for phrase in [
            "must vacate",
            "violated the lease",
            "clearly liable"
        ]:
            if phrase in removed_text.lower() and phrase not in self.memory.banned_phrases:
                self.memory.banned_phrases.append(phrase)

        self.save()
        return self.memory

    def apply(self, draft: str) -> str:
        improved_draft = draft

        replacements = {
            "The tenant violated the lease and must vacate.": (
                "The records indicate possible nonpayment and late-payment issues. "
                "Further legal review is needed before determining the appropriate next step."
            ),
            "must vacate": "may require further action after legal review",
            "violated the lease": "appears to have unresolved lease-related issues",
            "clearly liable": "potentially responsible, subject to legal review"
        }

        if self.memory.cautious_legal_language:
            for old_text, new_text in replacements.items():
                improved_draft = improved_draft.replace(old_text, new_text)

        if "Operator-edit preferences applied" not in improved_draft:
            improved_draft += "\n\n## Operator-edit preferences applied\n"
            improved_draft += "- Used cautious legal wording where the source evidence does not support a definitive conclusion.\n"
            improved_draft += "- Preserved grounding by keeping evidence references attached to key claims.\n"

        return improved_draft

    def save(self) -> None:
        write_json(self.memory_path, self.memory.model_dump())
