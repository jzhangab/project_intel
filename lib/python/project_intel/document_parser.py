"""
Parsers for Excel, PowerPoint, Word, and PDF documents.
Returns structured text content suitable for LLM processing.
"""

import io
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ParsedDocument:
    filename: str
    file_type: str
    content: str
    metadata: dict


def parse_document(filename: str, file_bytes: bytes) -> ParsedDocument:
    """Route to the appropriate parser based on file extension."""
    ext = os.path.splitext(filename.lower())[1]
    if ext == ".pdf":
        return _parse_pdf(filename, file_bytes)
    elif ext in (".docx", ".doc"):
        return _parse_word(filename, file_bytes)
    elif ext in (".pptx", ".ppt"):
        return _parse_powerpoint(filename, file_bytes)
    elif ext in (".xlsx", ".xls"):
        return _parse_excel(filename, file_bytes)
    else:
        return ParsedDocument(
            filename=filename,
            file_type=ext,
            content=f"[Unsupported file type: {ext}]",
            metadata={},
        )


def _parse_pdf(filename: str, file_bytes: bytes) -> ParsedDocument:
    import pdfplumber

    text_parts = []
    metadata = {}
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        metadata["pages"] = len(pdf.pages)
        if pdf.metadata:
            metadata.update({k: v for k, v in pdf.metadata.items() if v})
        for i, page in enumerate(pdf.pages, 1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                text_parts.append(f"[Page {i}]\n{page_text}")
            # Extract tables if present
            tables = page.extract_tables()
            for t_idx, table in enumerate(tables, 1):
                rows = [" | ".join(str(c) if c else "" for c in row) for row in table if any(row)]
                if rows:
                    text_parts.append(f"[Page {i} - Table {t_idx}]\n" + "\n".join(rows))

    return ParsedDocument(
        filename=filename,
        file_type="pdf",
        content="\n\n".join(text_parts),
        metadata=metadata,
    )


def _parse_word(filename: str, file_bytes: bytes) -> ParsedDocument:
    from docx import Document

    doc = Document(io.BytesIO(file_bytes))
    text_parts = []
    metadata = {}

    core_props = doc.core_properties
    if core_props.author:
        metadata["author"] = core_props.author
    if core_props.created:
        metadata["created"] = str(core_props.created)
    if core_props.modified:
        metadata["modified"] = str(core_props.modified)
    if core_props.title:
        metadata["title"] = core_props.title

    for para in doc.paragraphs:
        if para.text.strip():
            style = para.style.name if para.style else ""
            prefix = ""
            if "Heading" in style:
                level = "".join(filter(str.isdigit, style)) or "1"
                prefix = "#" * int(level) + " "
            text_parts.append(f"{prefix}{para.text}")

    for table in doc.tables:
        rows = []
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            rows.append(" | ".join(cells))
        if rows:
            text_parts.append("[Table]\n" + "\n".join(rows))

    return ParsedDocument(
        filename=filename,
        file_type="docx",
        content="\n".join(text_parts),
        metadata=metadata,
    )


def _parse_powerpoint(filename: str, file_bytes: bytes) -> ParsedDocument:
    from pptx import Presentation

    prs = Presentation(io.BytesIO(file_bytes))
    text_parts = []
    metadata = {"slides": len(prs.slides)}

    core_props = prs.core_properties
    if core_props.author:
        metadata["author"] = core_props.author
    if core_props.created:
        metadata["created"] = str(core_props.created)
    if core_props.title:
        metadata["title"] = core_props.title

    for slide_num, slide in enumerate(prs.slides, 1):
        slide_texts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    text = para.text.strip()
                    if text:
                        slide_texts.append(text)
            # Extract table data from slides
            if shape.has_table:
                for row in shape.table.rows:
                    cells = [cell.text.strip() for cell in row.cells]
                    slide_texts.append(" | ".join(cells))
        if slide_texts:
            text_parts.append(f"[Slide {slide_num}]\n" + "\n".join(slide_texts))

    return ParsedDocument(
        filename=filename,
        file_type="pptx",
        content="\n\n".join(text_parts),
        metadata=metadata,
    )


def _parse_excel(filename: str, file_bytes: bytes) -> ParsedDocument:
    import pandas as pd

    text_parts = []
    metadata = {}

    xl = pd.ExcelFile(io.BytesIO(file_bytes))
    metadata["sheets"] = xl.sheet_names

    for sheet_name in xl.sheet_names:
        df = xl.parse(sheet_name)
        # Drop entirely empty rows/columns
        df = df.dropna(how="all").dropna(axis=1, how="all")
        if df.empty:
            continue
        # Represent as readable markdown-style table (truncate very large sheets)
        rows_preview = min(len(df), 200)
        if len(df) > 200:
            note = f"[Showing first 200 of {len(df)} rows]"
        else:
            note = ""
        table_str = df.head(rows_preview).to_string(index=False, max_colwidth=100)
        entry = f"[Sheet: {sheet_name}]"
        if note:
            entry += f"\n{note}"
        entry += f"\n{table_str}"
        text_parts.append(entry)

    return ParsedDocument(
        filename=filename,
        file_type="xlsx",
        content="\n\n".join(text_parts),
        metadata=metadata,
    )
