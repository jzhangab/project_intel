"""
Document intelligence agent.
Processes each document with Claude, extracting structured insights into memory.
Handles large documents via chunking and streaming.
"""

import anthropic
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


def analyze_document(
    client: anthropic.Anthropic,
    doc: ParsedDocument,
    memory: MemoryStore,
) -> DocumentRecord:
    """
    Send a document to Claude for structured analysis.
    For large documents, processes in chunks and merges the results.
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

        full_response = ""
        with client.messages.stream(
            model="claude-opus-4-7",
            max_tokens=4096,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            cache_control={"type": "ephemeral"},
        ) as stream:
            for text in stream.text_stream:
                full_response += text

        chunk_summaries.append(full_response.strip())

    # If multiple chunks, merge them with a second pass
    if len(chunk_summaries) > 1:
        merge_prompt = f"""You analyzed a large document ({doc.filename}) in {len(chunk_summaries)} parts.
Here are the extracted insights from each part:

{chr(10).join(f"Part {i+1}:{chr(10)}{s}" for i, s in enumerate(chunk_summaries))}

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

        full_response = ""
        with client.messages.stream(
            model="claude-opus-4-7",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": merge_prompt}],
        ) as stream:
            for text in stream.text_stream:
                full_response += text
        final_json_str = full_response.strip()
    else:
        final_json_str = chunk_summaries[0]

    # Parse the JSON response
    import json
    import re
    from datetime import datetime

    # Extract JSON if wrapped in markdown code block
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", final_json_str, re.DOTALL)
    if json_match:
        final_json_str = json_match.group(1)

    try:
        extracted = json.loads(final_json_str)
    except json.JSONDecodeError:
        # Fallback: use the raw response as the summary
        extracted = {
            "summary": final_json_str[:500],
            "key_topics": [],
            "decisions": [],
            "technologies": [],
            "dates_mentioned": [],
            "people_mentioned": [],
        }

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
