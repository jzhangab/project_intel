"""
Final synthesis pass: uses all document summaries stored in memory
to produce a comprehensive project intelligence report via the Dataiku LLM Mesh.
"""

import json
import re

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


def generate_report(llm, memory: MemoryStore, project_name: str = "The Project") -> str:
    """
    Generate the final comprehensive project intelligence report.

    Args:
        llm: A Dataiku LLM handle obtained from dataiku.LLM("connection_id")
        memory: The populated agent memory store
        project_name: Human-readable project name for the report title
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

    print("Synthesizing final report (this may take a minute)...")
    report = _call_llm(llm, SYNTHESIS_SYSTEM, prompt, max_tokens=8000)
    print(f"Report generated: {len(report):,} characters")
    return report


def extract_global_insights(llm, memory: MemoryStore) -> dict:
    """
    Extract structured global metadata (timeline, decisions, cross-doc insights)
    and persist them back into memory.

    Args:
        llm: A Dataiku LLM handle obtained from dataiku.LLM("connection_id")
        memory: The populated agent memory store
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

    raw = _call_llm(llm, SYNTHESIS_SYSTEM, prompt, max_tokens=4096)

    # Strip markdown fences if present
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if match:
        raw = match.group(1)

    try:
        data = json.loads(raw.strip())
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
