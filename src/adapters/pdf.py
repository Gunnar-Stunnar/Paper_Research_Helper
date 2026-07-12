"""Adapter for downloading and extracting text from PDF files."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Union
import requests
import pypdf


class PDFAdapter:
    """Download PDFs from URLs and extract their text content."""

    def __init__(self, timeout: int = 30) -> None:
        self._timeout = timeout

    def from_url(self, url: str) -> str:
        """Download a PDF from *url* and return its full text."""
        response = requests.get(url, timeout=self._timeout)
        response.raise_for_status()
        return self._extract_text(io.BytesIO(response.content))

    def from_path(self, path: Union[str, Path]) -> str:
        """Read a local PDF file and return its full text."""
        with open(path, "rb") as fh:
            return self._extract_text(fh)

    @staticmethod
    def _extract_text(source: Union[io.BytesIO, "BinaryIO"]) -> str:  # noqa: F821
        reader = pypdf.PdfReader(source)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)
