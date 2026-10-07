"""
src/generation/incident_report.py - Incident Report Engine with dual-trigger generation gate.
Calls the LLM whenever diagnostic documents match above threshold OR a descriptive verbatim error message is provided.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from src.generation.prompt_builder import build_incident_prompt


class IncidentReportEngine:
    def __init__(self, monitoring_store: Any = None, search_engine: Any = None, llm_client: Any = None):
        self.monitoring_store = monitoring_store or []
        self.search_engine = search_engine
        self.llm_client = llm_client

    def generate_report(
        self,
        run_id: str,
        top_k: int = 3,
        threshold: float = 0.50,
        run_facts: Optional[Dict[str, Any]] = None,
        retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Generates an incident report, triggering LLM synthesis if reference docs or verbatim error output exists."""
        # Resolve facts if not explicitly provided
        if run_facts is None:
            found = [r for r in self.monitoring_store if str(getattr(r, "run_id", r.get("run_id") if isinstance(r, dict) else "")) == str(run_id)]
            if found:
                r = found[0]
                if isinstance(r, dict):
                    run_facts = r
                else:
                    run_facts = {
                        "run_id": r.run_id,
                        "pipeline_name": r.job_name,
                        "job_name": r.job_name,
                        "task_name": r.tasks[0].task_key if r.tasks else "N/A",
                        "duration_seconds": r.duration_seconds or 0,
                        "error_message": r.error_message or "",
                        "result_state": r.result_state.value,
                    }
            else:
                run_facts = {"run_id": run_id, "error_message": ""}

        # Retrieve reference chunks if search engine is available and not provided
        if retrieved_chunks is None and self.search_engine is not None:
            try:
                retrieved_chunks = self.search_engine.search(f"Why did run {run_id} fail?", top_k=top_k, min_score=threshold)
            except Exception:
                retrieved_chunks = []
        elif retrieved_chunks is None:
            retrieved_chunks = []

        # Filter retrieved documents by similarity threshold
        valid_evidence = [
            doc for doc in retrieved_chunks
            if (doc.get("score", 0) if isinstance(doc, dict) else getattr(doc, "score", 0)) >= threshold
        ]

        # Determine if we have sufficient grounds to run the LLM
        has_linked_docs = bool(valid_evidence)
        has_verbatim_error = bool(run_facts.get("error_message") and str(run_facts["error_message"]).strip())

        llm_used = False
        if (has_linked_docs or has_verbatim_error) and self.llm_client is not None:
            # Build prompt combining verified facts and reference evidence
            prompt = build_incident_prompt(run_facts=run_facts, evidence=valid_evidence)
            try:
                # Support both generate(prompt) and generate(system, user) interfaces
                if hasattr(self.llm_client, "generate"):
                    try:
                        resp = self.llm_client.generate(prompt)
                        llm_explanation = resp.text if hasattr(resp, "text") else str(resp)
                    except TypeError:
                        resp = self.llm_client.generate(
                            system="You are an expert Data Reliability Engineer analyzing a Databricks pipeline failure.",
                            user=prompt,
                        )
                        llm_explanation = resp.text if hasattr(resp, "text") else str(resp)
                    llm_used = True
                else:
                    llm_explanation = "LLM client not configured."
            except Exception as exc:
                llm_explanation = f"LLM generation failed: {exc}"
        else:
            llm_explanation = (
                "No explanation generated: verified facts alone or available evidence "
                "were insufficient to establish an explanation."
            )

        report_markdown = f"""# Incident Report: Run {run_facts.get('run_id')}

## Verified Facts
- **Run ID**: {run_facts.get('run_id')}
- **Job / Pipeline**: {run_facts.get('pipeline_name', run_facts.get('job_name', 'N/A'))}
- **Duration**: {run_facts.get('duration_seconds', 0)}s
- **Verbatim Error**: `{run_facts.get('error_message', '(none)')}`

## Analysis & Remediation
{llm_explanation}
"""

        return {
            "status": "success",
            "run_id": run_id,
            "report_markdown": report_markdown.strip(),
            "retrieved_evidence": valid_evidence,
            "llm_used": llm_used,
        }
