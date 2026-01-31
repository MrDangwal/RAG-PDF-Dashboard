from __future__ import annotations

from typing import Iterable
from PyPDF2 import PdfReader


def read_pdfs(pdf_files: Iterable) -> str:
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""
            text += page_text
    return text
