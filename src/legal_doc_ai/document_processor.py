
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import cv2
import fitz
import numpy as np
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

from .schemas import PageRecord, StructuredFields
from .utils import stable_doc_id


SUPPORTED_EXTENSIONS = {
    ".txt", ".md", ".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"
}


class DocumentProcessor:
    def __init__(self, min_pdf_text_chars: int = 40):
        self.min_pdf_text_chars = min_pdf_text_chars

    def process_directory(
        self,
        input_dir: str | Path
    ) -> Tuple[List[PageRecord], List[StructuredFields]]:
        input_dir = Path(input_dir)

        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

        files = [
            file for file in input_dir.rglob("*")
            if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        if not files:
            raise ValueError(f"No supported documents found in {input_dir}")

        all_pages = []
        all_structured_fields = []

        for file_path in sorted(files):
            pages = self.process_file(file_path)
            all_pages.extend(pages)

            combined_text = "\n".join(page.text for page in pages)
            structured = self.extract_structured_fields(
                doc_id=stable_doc_id(file_path),
                text=combined_text
            )
            all_structured_fields.append(structured)

        return all_pages, all_structured_fields

    def process_file(self, file_path: str | Path) -> List[PageRecord]:
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()

        if suffix in {".txt", ".md"}:
            return self._process_text(file_path)

        if suffix == ".pdf":
            return self._process_pdf(file_path)

        if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            return self._process_image(file_path)

        raise ValueError(f"Unsupported file type: {file_path}")

    def _process_text(self, file_path: Path) -> List[PageRecord]:
        text = file_path.read_text(encoding="utf-8", errors="replace")

        return [
            PageRecord(
                doc_id=stable_doc_id(file_path),
                source_path=str(file_path),
                page_number=1,
                text=self._normalize_text(text),
                extraction_method="text",
                ocr_confidence=None,
                warnings=[]
            )
        ]

    def _process_pdf(self, file_path: Path) -> List[PageRecord]:
        doc_id = stable_doc_id(file_path)
        records = []

        with fitz.open(file_path) as pdf:
            for page_index, page in enumerate(pdf, start=1):
                embedded_text = self._normalize_text(page.get_text("text") or "")

                if len(embedded_text.strip()) >= self.min_pdf_text_chars:
                    records.append(
                        PageRecord(
                            doc_id=doc_id,
                            source_path=str(file_path),
                            page_number=page_index,
                            text=embedded_text,
                            extraction_method="pdf_text",
                            ocr_confidence=None,
                            warnings=[]
                        )
                    )
                else:
                    ocr_text, confidence = self._ocr_pdf_page(file_path, page_index)

                    warnings = []
                    if confidence is not None and confidence < 55:
                        warnings.append(
                            "Low OCR confidence; this page may be partially unclear."
                        )

                    records.append(
                        PageRecord(
                            doc_id=doc_id,
                            source_path=str(file_path),
                            page_number=page_index,
                            text=self._normalize_text(ocr_text),
                            extraction_method="ocr_pdf_page",
                            ocr_confidence=confidence,
                            warnings=warnings
                        )
                    )

        return records

    def _process_image(self, file_path: Path) -> List[PageRecord]:
        text, confidence = self._ocr_image(file_path)

        warnings = []
        if confidence is not None and confidence < 55:
            warnings.append("Low OCR confidence; this image may be partially unclear.")

        return [
            PageRecord(
                doc_id=stable_doc_id(file_path),
                source_path=str(file_path),
                page_number=1,
                text=self._normalize_text(text),
                extraction_method="ocr_image",
                ocr_confidence=confidence,
                warnings=warnings
            )
        ]

    def _ocr_pdf_page(
        self,
        file_path: Path,
        page_number: int
    ) -> tuple[str, float | None]:
        images = convert_from_path(
            str(file_path),
            dpi=250,
            first_page=page_number,
            last_page=page_number
        )

        if not images:
            return "", None

        return self._ocr_pil_image(images[0])

    def _ocr_image(self, file_path: Path) -> tuple[str, float | None]:
        image = Image.open(file_path).convert("RGB")
        return self._ocr_pil_image(image)

    def _ocr_pil_image(self, image: Image.Image) -> tuple[str, float | None]:
        processed = self._preprocess_image_for_ocr(image)

        data = pytesseract.image_to_data(
            processed,
            output_type=pytesseract.Output.DICT,
            config="--psm 6"
        )

        words = []
        confidences = []

        for word, confidence in zip(data.get("text", []), data.get("conf", [])):
            word = word.strip()

            try:
                confidence_value = float(confidence)
            except ValueError:
                confidence_value = -1

            if word:
                words.append(word)

                if confidence_value >= 0:
                    confidences.append(confidence_value)

        average_confidence = (
            float(np.mean(confidences)) if confidences else None
        )

        return " ".join(words), average_confidence

    def _preprocess_image_for_ocr(self, image: Image.Image) -> Image.Image:
        array = np.array(image)
        gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, None, 20, 7, 21)

        thresholded = cv2.threshold(
            denoised,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]

        return Image.fromarray(thresholded)

    def extract_structured_fields(
        self,
        doc_id: str,
        text: str
    ) -> StructuredFields:
        fields = StructuredFields(doc_id=doc_id)

        patterns = {
            "landlord": r"(?:Landlord|From):\s*([^\n]+)",
            "tenant": r"(?:Tenant|To):\s*([^\n]+)",
            "property": r"(?:Property|Premises):\s*([^\n]+)",
            "lease_start": r"(?:Lease Start|lease dated):\s*([^\n\.]+)",
            "notice_date": r"(?:Date):\s*([^\n]+)",
            "monthly_rent": r"(?:Monthly Rent|rent in the amount of)[:\s]*\$?([0-9,]+)",
            "late_fee": r"(?:late fee(?: of)?|Late Payment).*?\$([0-9,]+)",
            "balance": r"(?:total balance.*?|balance currently shown is)\s*\$?([0-9,]+)",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)

            if not match:
                continue

            value = match.group(1).strip(" .;")

            if key in {"landlord", "tenant"}:
                fields.parties[key] = value
            elif key in {"lease_start", "notice_date"}:
                fields.dates[key] = value
            elif key in {"monthly_rent", "late_fee", "balance"}:
                fields.amounts[key] = "$" + value if not value.startswith("$") else value
            elif key == "property":
                fields.property_address = value

        issue_patterns = [
            ("unpaid rent", r"unpaid rent|rent .* remains unpaid|nonpayment"),
            ("late payment", r"late payment|late fee|not received by the fifth"),
            ("written notice", r"written notice|notice required|provided in writing"),
        ]

        for issue_name, pattern in issue_patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                fields.issues.append(issue_name)

        required_fields = [
            ("tenant", fields.parties.get("tenant")),
            ("landlord", fields.parties.get("landlord")),
            ("property_address", fields.property_address),
        ]

        for field_name, value in required_fields:
            if not value:
                fields.uncertainty_notes.append(
                    f"{field_name} not confidently extracted"
                )

        return fields

    def _normalize_text(self, text: str) -> str:
        text = text.replace("\x0c", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
