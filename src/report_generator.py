"""
Final synthesis pass: uses all document summaries stored in memory
to produce a comprehensive project intelligence report.
"""

import anthropic
from .memory import MemoryStore

SYNTHESIS_SYSTEM = """You are a principal technology consultant writing an executive-grade
project intelligence report. You have access to summaries of all project documents and
must synthesize them into a coherent, insightful narrative.

Your report must reconstruct:
1. The full history of the project — origins, phases, pivots
2. Key architectural and design decisions, with their rationale where available
3. The technology stack and how it evolved
4. Timeline of major milestones and events
5. Team structure and key stakeholders
6. Challenges encountered and how they were resolved
7. Current state of the project (as evidenced by documents)
8. Patterns, gaps, and observations across the document corpus

Write with clarity and structure. Use headers, bullet points, and tables where appropriate.
Base your analysis strictly on evidence from the documents. Note where information is inferred
vs. explicitly stated."""


def generate_report(
    client: anthropic.Anthropic,
    memory: MemoryStore,
    project_name: str = "The Project",
) -> str:
    """
    Generate the final comprehensive project intelligence report.
    Uses cached document summaries to keep the context efficient.
    """
    all_summaries = memory.get_all_summaries()
    stats = memory.get_stats()

    prompt = f"""You have analyzed {stats['total_documents']} documents from the project repository.
The documents span these file types: {', '.join(stats['file_types'])}.

Below are the extracted summaries and structured insights from every document:

{all_summaries}

---

Now write a comprehensive PROJECT INTELLIGENCE REPORT for: **{project_name}**

Structure your report with these sections:

# Project Intelligence Report: {project_name}

## Executive Summary
(3-5 paragraphs synthesizing the overall project story)

## Project History & Timeline
(Reconstruct the chronological story of the project based on all evidence)

## Technical Architecture & Design
(Describe the system design, key architectural decisions, and technical approach)

## Technology Stack
(Complete inventory of technologies, tools, frameworks, platforms)

## Key Decisions & Rationale
(Document the significant decisions made and their stated or inferred reasoning)

## Team & Stakeholders
(Who was involved, their roles, organizational structure)

## Challenges & Solutions
(Problems encountered, how they were addressed)

## Current State Assessment
(What can be concluded about the project's current status)

## Cross-Document Patterns & Observations
(Insights that only emerge when looking across all documents)

## Information Gaps & Uncertainties
(What is unclear, missing, or contradictory across documents)

Write a thorough, evidence-based report. This report will be used to onboard new team members
and brief executives who have not read the underlying documents."""

    print("Synthesizing final report...")
    report_parts = []

    with client.messages.stream(
        model="claude-opus-4-7",
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system=[
            {
                "type": "text",
                "text": SYNTHESIS_SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            report_parts.append(text)
            print(text, end="", flush=True)

    print()  # newline after streaming
    return "".join(report_parts)


def extract_global_insights(
    client: anthropic.Anthropic,
    memory: MemoryStore,
) -> dict:
    """
    Extract structured global metadata (timeline, decisions, insights) to persist in memory.
    """
    all_summaries = memory.get_all_summaries()

    prompt = f"""Based on the following document summaries from a technology project, extract
structured global metadata. Return a JSON object only.

{all_summaries}

Return JSON:
{{
  "global_themes": ["overarching theme or pattern across documents"],
  "project_timeline": [
    {{"date": "date or period", "event": "what happened", "source": "filename"}}
  ],
  "key_decisions": [
    {{"decision": "what was decided", "rationale": "why if known", "source": "filename"}}
  ],
  "cross_document_insights": ["insight that only emerges from multiple documents"]
}}"""

    full_response = ""
    with client.messages.stream(
        model="claude-opus-4-7",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            full_response += text

    import json
    import re

    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", full_response, re.DOTALL)
    if json_match:
        full_response = json_match.group(1)

    try:
        data = json.loads(full_response.strip())
    except json.JSONDecodeError:
        data = {
            "global_themes": [],
            "project_timeline": [],
            "key_decisions": [],
            "cross_document_insights": [],
        }

    memory.update_global_insights(
        themes=data.get("global_themes", []),
        timeline=data.get("project_timeline", []),
        decisions=data.get("key_decisions", []),
        insights=data.get("cross_document_insights", []),
    )
    return data
