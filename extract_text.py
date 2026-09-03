"""
extract_text.py — Text Extraction Layer for the Resume Parser project.

Goal: given a resume file (.pdf, .docx, or a scanned/image-based PDF),
return clean plain text ready for the downstream NLP/NER stage.

Design notes:
- PDFs are tried as text-based first (fast, high fidelity). If a page yields
  little or no text, we fall back to OCR (this handles scanned resumes
  without needing the caller to know in advance which type a file is).
- DOCX files are parsed for both paragraphs and tables (skills/dates are
  often laid out in tables).
- Every function returns an ExtractionResult with metadata (method used,
  per-page confidence for OCR, warnings) — useful for your evaluation report,
  since "how the text was obtained" affects downstream NER quality.

Dependencies:
    pip install pdfplumber python-docx pytesseract pdf2image --break-system-packages
    (system) tesseract-ocr and poppler-utils must be installed for OCR/pdf2image)
"""

from __future__ import annotations

from csv import reader
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import pdfplumber
import pytesseract
import easyocr
from docx import Document
from pdf2image import convert_from_path

# --- Tunables -----------------------------------------------------------

# If a PDF page has fewer than this many characters of extracted text,
# treat it as image-only and OCR it instead.
MIN_CHARS_PER_PAGE = 20

# DPI used when rasterizing PDF pages for OCR. Higher = more accurate,
# slower. 200-300 is a good range for resume text.
OCR_DPI = 300 


# --- Result container -----------------------------------------------------

@dataclass # dataclass decorator automatically generates init, repr, and other methods
class ExtractionResult: # container for the result of a text extraction operation
    source_path: str # path to the source file from which text was extracted
    text: str # the extracted text content
    method: str  # "pdf_text" | "pdf_ocr" | "pdf_mixed" | "docx" | "image_ocr"
    pages: range = field(default=range(1,4)) # range of page numbers processed (for multi-page documents) 
    warnings: list[str] = field(default=list) # list of warnings generated during extraction (e.g., pages with no text, OCR failures)

    def to_dict(self) -> dict:
        return asdict(self) #return the fields of a dataclass instance as a new dictionary mapping field names to field values


class ExtractionError(Exception):
    """Raised when a file cannot be processed at all."""


# --- PDF extraction ---------------------------------------------------------

def extract_from_pdf(path: str | Path) -> ExtractionResult:
    """
    Extract text from a PDF. Tries the embedded text layer per page first;
    any page with too little text is OCR'd instead. This means a single
    PDF can be "pdf_mixed" if only some pages are scanned images.
    """
    path = Path(path)
    warnings: list[str] = []
    page_texts: list[str] = []
    methods_used: set[str] = set()

    with pdfplumber.open(path) as pdf:
        num_pages = len(pdf.pages)
        for i, page in enumerate(pdf.pages):
            raw = page.extract_text() or ""
            if len(raw.strip()) >= MIN_CHARS_PER_PAGE:
                page_texts.append(raw)
                methods_used.add("pdf_text")
            else:
                ocr_text = _ocr_pdf_page(path, i)
                if not ocr_text.strip():
                    warnings.append(f"page {i + 1}: no text extracted (text layer and OCR both empty)")
                page_texts.append(ocr_text)
                methods_used.add("pdf_ocr")

    if methods_used == {"pdf_text"}:
        method = "pdf_text"
    elif methods_used == {"pdf_ocr"}:
        method = "pdf_ocr"
    elif methods_used:
        method = "pdf_mixed"
    else:
        method = "pdf_text"
        warnings.append("empty PDF: no pages found")

    full_text = _clean_text("\n".join(page_texts))
    return ExtractionResult(
        source_path=str(path), text=full_text, method=method,
        pages=num_pages, warnings=warnings,
    )


def _ocr_pdf_page(pdf_path: Path, page_index: int) -> str: #extract text from a single page of a PDF using OCR
    """Rasterize a single PDF page and run Tesseract OCR on it."""
    images = convert_from_path(
        str(pdf_path), dpi=OCR_DPI,
        first_page=page_index + 1, last_page=page_index + 1,
    )
     
    if not images:
        return ""
    return pytesseract.image_to_string(images[0])


# --- DOCX extraction ---------------------------------------------------------

def extract_from_docx(path: str | Path) -> ExtractionResult:
    """
    Extract text from a .docx file: paragraphs in order, plus any tables
    (resumes frequently put skills or dates in table layouts, which
    python-docx's default paragraph iteration skips over).
    """
    path = Path(path)
    doc = Document(str(path))
    warnings: list[str] = []

    chunks: list[str] = []
    for para in doc.paragraphs:
        if para.text.strip():
            chunks.append(para.text.strip())

    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                chunks.append(" | ".join(cells))

    if not chunks:
        warnings.append("no extractable text found in document body or tables")

    full_text = _clean_text("\n".join(chunks))
    return ExtractionResult(
        source_path=str(path), text=full_text, method="docx",
        pages=1, warnings=warnings,
    )

# --- Standalone image extraction (e.g. a resume photographed / scanned as .png/.jpg/ .jpeg) ---

def extract_from_image(path: str | Path) -> ExtractionResult:
    from PIL import Image

    path = Path(path)
    warnings: list[str] = []
    img = Image.open(path)
    text = pytesseract.image_to_string(img)
    if not text.strip():
        warnings.append("OCR returned no text — check image quality/resolution")
    return ExtractionResult(
        source_path=str(path), text=_clean_text(text), method="image_ocr",
        pages=1, warnings=warnings,
    )


# --- Cleaning ---------------------------------------------------------------

import re

_CID_ARTIFACT_RE = re.compile(r"\(cid:\d+\)")


def _clean_text(text: str) -> str:
    """Light normalization: collapse whitespace, drop blank-heavy noise,
    keep line breaks (section detection downstream relies on them)."""
    # pdfplumber sometimes can't map a glyph (often custom bullet fonts) and
    # emits a raw "(cid:127)"-style token instead of the character. 
    # Bullets aren't semantically useful for NER, so we just drop these.
    text = _CID_ARTIFACT_RE.sub("", text)
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    # collapse runs of internal whitespace within each line
    lines = [" ".join(line.split()) for line in lines]
    return "\n".join(lines)


# --- Dispatcher ---------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".png", ".jpg", ".jpeg"}

def extract(path: str | Path) -> ExtractionResult:
    """Route a file to the right extractor based on its extension."""
    path = Path(path)
    if not path.exists():
        raise ExtractionError(f"File not found: {path}")

    ext = path.suffix.lower() #get the file extension in lowercase
    if ext == ".pdf":
        return extract_from_pdf(path) #use the extract_from_pdf function to extract text if extension is .pdf
    elif ext == ".docx":
        return extract_from_docx(path) #use the extract_from_docx function to extract text if extension is .docx
    elif ext in {".png", ".jpg", ".jpeg"}:
        return extract_from_image(path) #use the extract_from_image function to extract text if extension is .png, .jpg, or .jpeg
    else:
        raise ExtractionError(
            f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )


# --- CLI ---------------------------------------------------------------------
input_dir = r"C:\Users\25471\Desktop\Resume Parser Project\resume_samples"
output_dir = r"C:\Users\25471\Desktop\Resume Parser Project\extracted_outputs"

def process_directory(input_dir: str | Path, output_dir: str | Path) -> None:
    """
    Batch mode: process every supported resume file in input_dir, writing one
    JSON file per resume to output_dir (text + extraction metadata). This is
    the shape the downstream NER stage should read from.
    """
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = [f for f in input_dir.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not files:
        print(f"No supported files found in {input_dir}")
        return

    for f in files:
        try:
            result = extract(f)
            out_path = output_dir / f"{f.stem}.json"
            out_path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding='utf-8')
            status = "OK" if not result.warnings else f"OK (warnings: {result.warnings})"
            print(f"[{result.method:9s}] {f.name} -> {out_path.name}  {status}")
        except ExtractionError as e:
            print(f"[FAILED] {f.name}: {e}")


output = process_directory(input_dir, output_dir)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        process_directory(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 2:
        result = extract(sys.argv[1])
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        print("Usage:")
        print("  Single file:  python extract_text.py <resume_file>")
        print("  Batch mode:   python extract_text.py <input_dir> <output_dir>")
        sys.exit(1) #exit with error code 1 for incorrect usage
