"""PDF bytes -> cleaned plain text."""
import re

import fitz  # PyMuPDF

_REFS = re.compile(r"\n\s*(references|bibliography)\s*\n", re.IGNORECASE)
_HYPHEN = re.compile(r"(\w)-\n(\w)")
_PAGE_NUM = re.compile(r"^\s*\d{1,3}\s*$", re.MULTILINE)
_ARXIV_STAMP = re.compile(r"^arXiv:\d{4}\.\d{4,5}.*$", re.MULTILINE)
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f�]")


def clean(text: str) -> str:
    text = _CTRL.sub("", text)
    text = _HYPHEN.sub(r"\1\2", text)
    text = _ARXIV_STAMP.sub("", text)
    text = _PAGE_NUM.sub("", text)
    # drop the reference list (last heading, only if in the back half)
    matches = list(_REFS.finditer(text))
    if matches and matches[-1].start() > len(text) * 0.5:
        text = text[: matches[-1].start()]
    # join wrapped lines inside paragraphs, keep paragraph breaks
    paras = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text)]
    return "\n\n".join(p for p in paras if len(p) > 30)


def pdf_to_text(data: bytes) -> str:
    with fitz.open(stream=data, filetype="pdf") as doc:
        return "\n".join(page.get_text("text") for page in doc)
