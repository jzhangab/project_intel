"""
Top-level pipeline for Project Intelligence.

Single entry point: call run_pipeline() from a notebook or recipe.
All orchestration, progress reporting, and error handling lives here.
"""

import os
from dataclasses import dataclass

import dataiku
from tqdm import tqdm

from project_intel.document_parser import parse_document
from project_intel.memory import MemoryStore
from project_intel.agent import analyze_document
from project_intel.report_generator import extract_global_insights, generate_report

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls"}


@dataclass
class PipelineResult:
    report: str
    report_path: str
    memory_path: str
    stats: dict
    errors: list[dict]

    def summary(self) -> str:
        lines = [
            "=== Project Intelligence — Run Summary ===",
            f"Documents processed : {self.stats.get('total_documents', 0)}",
            f"File types          : {', '.join(self.stats.get('file_types', []))}",
            f"Technologies found  : {self.stats.get('total_technologies', 0)}",
            f"Stakeholders found  : {self.stats.get('total_stakeholders', 0)}",
            f"Processing errors   : {len(self.errors)}",
            f"Report saved to     : {self.report_path}",
            f"Memory saved to     : {self.memory_path}",
        ]
        if self.errors:
            lines.append("\nFailed files:")
            for e in self.errors:
                lines.append(f"  ✗ {e['file']}: {e['error']}")
        return "\n".join(lines)


def run_pipeline(
    llm_connection_id: str,
    folder_name: str = "project_documents",
    project_name: str = "My Technology Project",
    memory_path: str = "agent_memory.json",
    report_output_path: str = "project_intel_report.md",
    skip_already_processed: bool = True,
    output_folder_name: str = None,
) -> PipelineResult:
    """
    Run the full Project Intelligence pipeline.

    Reads all supported documents from `folder_name`, processes each through
    the LLM agent, builds a persistent memory of the corpus, extracts cross-document
    insights, and produces a comprehensive project intelligence report.

    Args:
        llm_connection_id:      Dataiku LLM connection name (Admin > Connections > LLM)
        folder_name:            Dataiku managed folder containing project documents
        project_name:           Human-readable project name used in the report title
        memory_path:            Local path to persist the agent memory JSON
        report_output_path:     Local path to write the final Markdown report
        skip_already_processed: If True, resume from previous run without reprocessing docs
        output_folder_name:     Optional Dataiku managed folder to also write the report into

    Returns:
        PipelineResult with the report text, paths, stats, and any errors
    """
    llm = dataiku.LLM(llm_connection_id)
    folder = dataiku.Folder(folder_name)

    # ── Discover documents ────────────────────────────────────────────────────
    all_paths = folder.list_paths_in_partition()
    document_paths = [
        p for p in all_paths
        if os.path.splitext(p.lower())[1] in SUPPORTED_EXTENSIONS
    ]
    print(f"Found {len(document_paths)} supported documents in '{folder_name}'.")

    # ── Memory store ──────────────────────────────────────────────────────────
    memory = MemoryStore(memory_path=memory_path)
    memory.memory.project_name = project_name
    memory.save()

    to_process = (
        [p for p in document_paths if not memory.is_processed(os.path.basename(p))]
        if skip_already_processed
        else document_paths
    )
    already_done = len(document_paths) - len(to_process)
    if already_done:
        print(f"Skipping {already_done} already-processed documents (resume mode).")
    print(f"Processing {len(to_process)} documents...")

    # ── Document analysis loop ────────────────────────────────────────────────
    errors = []
    for path in tqdm(to_process, desc="Analyzing documents"):
        filename = os.path.basename(path)
        print(f"\n→ {filename}")
        try:
            with folder.get_download_stream(path) as stream:
                file_bytes = stream.read()

            parsed = parse_document(filename, file_bytes)

            if not parsed.content.strip():
                print(f"  ⚠ No text content extracted, skipping.")
                continue

            print(f"  {len(parsed.content):,} characters extracted")
            record = analyze_document(llm, parsed, memory)
            print(f"  ✓ {record.summary[:100]}...")

        except Exception as exc:
            print(f"  ✗ {exc}")
            errors.append({"file": filename, "error": str(exc)})

    # ── Cross-document synthesis ──────────────────────────────────────────────
    print("\nExtracting cross-document patterns...")
    insights = extract_global_insights(llm, memory)
    print(f"  Themes: {len(insights.get('global_themes', []))}, "
          f"Timeline events: {len(insights.get('project_timeline', []))}, "
          f"Key decisions: {len(insights.get('key_decisions', []))}")

    # ── Final report ──────────────────────────────────────────────────────────
    print("\nGenerating project intelligence report...")
    report = generate_report(llm, memory, project_name=project_name)

    with open(report_output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report written to: {report_output_path} ({len(report):,} chars)")

    # ── Optional: write to Dataiku output folder ──────────────────────────────
    if output_folder_name:
        output_folder = dataiku.Folder(output_folder_name)
        with output_folder.get_writer("project_intel_report.md") as writer:
            writer.write(report.encode("utf-8"))
        print(f"Report also written to Dataiku folder: {output_folder_name}")

    return PipelineResult(
        report=report,
        report_path=report_output_path,
        memory_path=memory_path,
        stats=memory.get_stats(),
        errors=errors,
    )
