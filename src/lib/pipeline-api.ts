const API_BASE = "http://localhost:8000/api";

export interface PipelineRun {
  run_id: string | number;
  job_id?: string | number;
  pipeline_name?: string;
  job_name?: string;
  status: string;
  lifecycle_state?: string;
  start_time: string | null;
  end_time?: string | null;
  duration_seconds: number;
  duration_formatted?: string;
  error_message?: string;
  tasks_count?: number;
  source?: string;
}

export interface IncidentReportResponse {
  status: string;
  run_id: string | number;
  job_id?: string | number;
  job_name?: string;
  result_state?: string;
  report_markdown: string;
  has_prior_occurrences?: boolean;
  evidence_count?: number;
  llm_used?: boolean;
}

export interface ChatResponse {
  route: string;
  reply: string;
  llm_used?: boolean;
  resolved_run_id?: string | number | null;
  warnings?: string[];
  evidence?: Array<{
    chunk_id: string;
    text: string;
    score?: number;
  }>;
}

export async function fetchRuns(source?: "live" | "mock"): Promise<PipelineRun[]> {
  const url = source ? `${API_BASE}/runs?source=${source}` : `${API_BASE}/runs`;
  const res = await fetch(url);
  if (!res.ok) throw new Error("Failed to load runs");
  const data = await res.json();
  return data.runs;
}

export async function refreshLiveRuns(): Promise<{ status: string; count: number }> {
  const res = await fetch(`${API_BASE}/runs/refresh`, { method: "POST" });
  if (!res.ok) throw new Error("Failed to refresh runs from Databricks");
  return await res.json();
}

export async function fetchIncidentReport(
  runId: string | number,
  similarityThreshold = 0.5
): Promise<IncidentReportResponse> {
  const res = await fetch(
    `${API_BASE}/incident-report/${runId}?similarity_threshold=${similarityThreshold}`
  );
  if (!res.ok) throw new Error(`Failed to load incident report for run ${runId}`);
  return await res.json();
}

export async function sendChatMessage(
  query: string,
  threshold = 0.5
): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, similarity_threshold: threshold }),
  });
  if (!res.ok) throw new Error("Failed to process question");
  return await res.json();
}
