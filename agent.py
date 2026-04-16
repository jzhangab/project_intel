"""
Document intelligence agent.
Processes each document through the Dataiku LLM Mesh, extracting structured
insights into memory. Handles large documents via chunking.
"""

import json
import re
from datetime import datetime

from .document_parser import ParsedDocument
from .memory import DocumentRecord, MemoryStore

# Chunk size in characters — large enough for context but safe for token limits
CHUNK_SIZE = 60_000

SYSTEM_PROMPT = """You are a senior technology project analyst specializing in reconstructing
project history and intelligence from documentary evidence. Your role is to carefully read
documents and extract structured, factual insights about the technology project they describe.

You focus on:
- Project decisions and their rationale
- Technical architecture and design choices
- Timeline and key milestones
- Technologies, tools, and platforms used
- Stakeholders and team structure
- Problems encountered and how they were resolved
- Evolution of the project over time

Be precise, factual, and thorough. When uncertain, note the uncertainty.
Do not fabricate details not present in the document."""


def _chunk_content(content: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """Split content into overlapping chunks to avoid losing context at boundaries."""
    if len(content) <= chunk_size:
        return [content]
    chunks = []
    overlap = chunk_size // 10
    start = 0
    while start < len(content):
        end = min(start + chunk_size, len(content))
        chunks.append(content[start:end])
        if end == len(content):
            break
        start = end - overlap
    return chunks


def _call_llm(llm, system: str, user: str, max_tokens: int = 4096) -> str:
    """Execute a single LLM call via the Dataiku LLM Mesh and return the response text."""
    completion = llm.new_completion()
    completion.with_message(system, role="system")
    completion.with_message(user, role="user")
    completion.settings["max_tokens"] = max_tokens
    resp = completion.execute()
    if not resp.success:
        raise RuntimeError(f"LLM call failed: {resp}")
    return resp.text.strip()


def _parse_json_response(raw: str) -> dict:
    """Extract and parse a JSON object from an LLM response, handling markdown fences."""
    # Strip markdown code block if present
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if match:
        raw = match.group(1)
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        return {
            "summary": raw[:500],
            "key_topics": [],
            "decisions": [],
            "technologies": [],
            "dates_mentioned": [],
            "people_mentioned": [],
        }


def analyze_document(llm, doc: ParsedDocument, memory: MemoryStore) -> DocumentRecord:
    """
    Send a document to the LLM for structured analysis via Dataiku LLM Mesh.
    For large documents, processes in chunks then merges results.

    Args:
        llm: A Dataiku LLM handle obtained from dataiku.LLM("connection_id")
        doc: Parsed document content
        memory: The agent memory store to persist results into
    """
    chunks = _chunk_content(doc.content)
    chunk_summaries = []

    for i, chunk in enumerate(chunks):
        chunk_label = f" (part {i+1}/{len(chunks)})" if len(chunks) > 1 else ""
        prompt = f"""Analyze the following document content and extract structured information.

Document: {doc.filename}{chunk_label}
File type: {doc.file_type}
Metadata: {doc.metadata}

---DOCUMENT CONTENT START---
{chunk}
---DOCUMENT CONTENT END---

Return a JSON object with exactly these fields:
{{
  "summary": "2-4 sentence summary of what this document covers",
  "key_topics": ["topic1", "topic2", ...],
  "decisions": ["decision or conclusion stated in document", ...],
  "technologies": ["tech/tool/platform mentioned", ...],
  "dates_mentioned": ["any dates or time references", ...],
  "people_mentioned": ["names of people or roles mentioned", ...]
}}

Return ONLY the JSON object, no other text."""

        raw = _call_llm(llm, SYSTEM_PROMPT, prompt, max_tokens=4096)
        chunk_summaries.append(raw)

    # If multiple chunks, do a merge pass
    if len(chunk_summaries) > 1:
        parts_text = "\n\n".join(
            f"Part {i+1}:\n{s}" for i, s in enumerate(chunk_summaries)
        )
        merge_prompt = f"""You analyzed a large document ({doc.filename}) in {len(chunk_summaries)} parts.
Here are the extracted insights from each part:

{parts_text}

Merge all parts into a single unified JSON object:
{{
  "summary": "comprehensive 3-5 sentence summary of the entire document",
  "key_topics": ["deduplicated list of all topics"],
  "decisions": ["deduplicated list of all decisions/conclusions"],
  "technologies": ["deduplicated list of all technologies"],
  "dates_mentioned": ["deduplicated list of all dates"],
  "people_mentioned": ["deduplicated list of all people/roles"]
}}

Return ONLY the JSON object."""
        final_raw = _call_llm(llm, SYSTEM_PROMPT, merge_prompt, max_tokens=4096)
    else:
        final_raw = chunk_summaries[0]

    extracted = _parse_json_response(final_raw)

    record = DocumentRecord(
        filename=doc.filename,
        file_type=doc.file_type,
        processed_at=datetime.utcnow().isoformat(),
        summary=extracted.get("summary", ""),
        key_topics=extracted.get("key_topics", []),
        decisions=extracted.get("decisions", []),
        technologies=extracted.get("technologies", []),
        dates_mentioned=extracted.get("dates_mentioned", []),
        people_mentioned=extracted.get("people_mentioned", []),
        metadata=doc.metadata,
    )

    memory.add_document(record)
    return record
