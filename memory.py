"""
Persistent memory system for tracking processed documents and extracted insights.
Stores document summaries, key entities, decisions, and timeline events.
"""

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class DocumentRecord:
    filename: str
    file_type: str
    processed_at: str
    summary: str
    key_topics: list[str]
    decisions: list[str]
    technologies: list[str]
    dates_mentioned: list[str]
    people_mentioned: list[str]
    metadata: dict


@dataclass
class AgentMemory:
    project_name: str
    created_at: str
    last_updated: str
    documents: dict[str, DocumentRecord] = field(default_factory=dict)
    global_themes: list[str] = field(default_factory=list)
    project_timeline: list[dict] = field(default_factory=list)
    key_decisions: list[dict] = field(default_factory=list)
    technologies_used: list[str] = field(default_factory=list)
    stakeholders: list[str] = field(default_factory=list)
    cross_document_insights: list[str] = field(default_factory=list)


class MemoryStore:
    def __init__(self, memory_path: str = "agent_memory.json"):
        self.memory_path = memory_path
        self.memory: AgentMemory = self._load_or_create()

    def _load_or_create(self) -> AgentMemory:
        if os.path.exists(self.memory_path):
            with open(self.memory_path, "r") as f:
                data = json.load(f)
            docs = {
                k: DocumentRecord(**v) for k, v in data.get("documents", {}).items()
            }
            data["documents"] = docs
            return AgentMemory(**data)
        return AgentMemory(
            project_name="Unknown Project",
            created_at=datetime.utcnow().isoformat(),
            last_updated=datetime.utcnow().isoformat(),
        )

    def save(self):
        data = asdict(self.memory)
        with open(self.memory_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def add_document(self, record: DocumentRecord):
        self.memory.documents[record.filename] = record
        self.memory.last_updated = datetime.utcnow().isoformat()
        # Merge technologies into global list
        for tech in record.technologies:
            if tech not in self.memory.technologies_used:
                self.memory.technologies_used.append(tech)
        # Merge people into stakeholders
        for person in record.people_mentioned:
            if person not in self.memory.stakeholders:
                self.memory.stakeholders.append(person)
        self.save()

    def is_processed(self, filename: str) -> bool:
        return filename in self.memory.documents

    def get_all_summaries(self) -> str:
        """Return a condensed view of all processed document summaries for synthesis."""
        parts = []
        for fname, rec in self.memory.documents.items():
            parts.append(
                f"## {fname} ({rec.file_type})\n"
                f"**Summary:** {rec.summary}\n"
                f"**Key Topics:** {', '.join(rec.key_topics)}\n"
                f"**Decisions:** {'; '.join(rec.decisions) if rec.decisions else 'None noted'}\n"
                f"**Technologies:** {', '.join(rec.technologies) if rec.technologies else 'None noted'}\n"
                f"**Dates:** {', '.join(rec.dates_mentioned) if rec.dates_mentioned else 'None noted'}\n"
                f"**People:** {', '.join(rec.people_mentioned) if rec.people_mentioned else 'None noted'}"
            )
        return "\n\n".join(parts)

    def get_stats(self) -> dict:
        return {
            "total_documents": len(self.memory.documents),
            "file_types": list({r.file_type for r in self.memory.documents.values()}),
            "total_technologies": len(self.memory.technologies_used),
            "total_stakeholders": len(self.memory.stakeholders),
        }

    def update_global_insights(self, themes: list[str], timeline: list[dict],
                                decisions: list[dict], insights: list[str]):
        self.memory.global_themes = themes
        self.memory.project_timeline = timeline
        self.memory.key_decisions = decisions
        self.memory.cross_document_insights = insights
        self.memory.last_updated = datetime.utcnow().isoformat()
        self.save()
