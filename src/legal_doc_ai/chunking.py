
from __future__ import annotations

from typing import List

from .schemas import Chunk, PageRecord


class TextChunker:
    def __init__(
        self,
        chunk_size_chars: int = 900,
        overlap_chars: int = 150
    ):
        if overlap_chars >= chunk_size_chars:
            raise ValueError("overlap_chars must be smaller than chunk_size_chars")

        self.chunk_size_chars = chunk_size_chars
        self.overlap_chars = overlap_chars

    def chunk_pages(self, pages: list[PageRecord]) -> List[Chunk]:
        chunks = []

        for page in pages:
            text = page.text.strip()

            if not text:
                continue

            start = 0
            chunk_index = 0

            while start < len(text):
                end = min(start + self.chunk_size_chars, len(text))
                window = text[start:end]

                if end < len(text):
                    sentence_end = max(window.rfind("."), window.rfind("\n"))

                    if sentence_end > int(self.chunk_size_chars * 0.55):
                        end = start + sentence_end + 1
                        window = text[start:end]

                chunk_id = f"{page.doc_id}:p{page.page_number}:c{chunk_index}"

                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        doc_id=page.doc_id,
                        source_path=page.source_path,
                        page_number=page.page_number,
                        text=window.strip(),
                        start_char=start,
                        end_char=end
                    )
                )

                if end == len(text):
                    break

                start = max(0, end - self.overlap_chars)
                chunk_index += 1

        return chunks
