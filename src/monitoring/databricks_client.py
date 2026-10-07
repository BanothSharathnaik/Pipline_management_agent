import os
import requests
from typing import List, Dict, Optional
from datetime import datetime, timezone
from src.monitoring.models import RunRecord  # Adjust import path if needed

class DatabricksClient:
    def __init__(self, host: Optional[str] = None, token: Optional[str] = None):
        self.host = (host or os.getenv("DATABRICKS_HOST", "")).rstrip("/")
        self.token = token or os.getenv("DATABRICKS_TOKEN", "")
        
        if not self.host or not self.token:
            raise ValueError("DATABRICKS_HOST and DATABRICKS_TOKEN must be set.")
            
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        # In-memory cache for job_id -> job_name mapping
        self._job_name_cache: Dict[int, str] = {}

    def _get_job_name(self, job_id: int) -> str:
        """Fetch job name from Databricks API with simple caching."""
        if job_id in self._job_name_cache:
            return self._job_name_cache[job_id]
            
        url = f"{self.host}/api/2.2/jobs/get"
        try:
            res = requests.get(url, headers=self.headers, params={"job_id": job_id}, timeout=10)
            if res.status_code == 200:
                name = res.json().get("settings", {}).get("name", f"Job #{job_id}")
                self._job_name_cache[job_id] = name
                return name
        except Exception:
            pass
            
        default_name = f"Job #{job_id}"
        self._job_name_cache[job_id] = default_name
        return default_name

    def fetch_recent_runs(self, limit: int = 20) -> List[RunRecord]:
        """Fetch recent job runs and map them to RunRecord models."""
        url = f"{self.host}/api/2.2/jobs/runs/list"
        params = {"limit": limit, "active_only": "false"}
        
        response = requests.get(url, headers=self.headers, params=params, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        runs_data = data.get("runs", [])
        
        records = []
        for run in runs_data:
            records.append(self._parse_run_record(run))
            
        return records

    def _parse_run_record(self, run: dict) -> RunRecord:
        run_id = str(run.get("run_id"))
        job_id = run.get("job_id")
        job_name = self._get_job_name(job_id) if job_id else "Unknown Job"
        
        state = run.get("state", {})
        life_cycle_state = state.get("life_cycle_state", "UNKNOWN")
        result_state = state.get("result_state")  # Can be None for active/queued runs
        state_message = state.get("state_message", "") or ""
        
        # Determine overall normalized status
        if life_cycle_state in ["QUEUED", "PENDING"]:
            status = "QUEUED"
        elif life_cycle_state in ["RUNNING", "TERMINATING"]:
            status = "RUNNING"
        elif life_cycle_state == "SKIPPED":
            status = "SKIPPED"
        elif life_cycle_state == "INTERNAL_ERROR":
            status = "FAILED"
        elif life_cycle_state == "TERMINATED":
            status = "SUCCESS" if result_state == "SUCCESS" else "FAILED"
        else:
            status = result_state or life_cycle_state

        start_time_ms = run.get("start_time", 0)
        end_time_ms = run.get("end_time", 0)
        
        start_time = datetime.fromtimestamp(start_time_ms / 1000.0, tz=timezone.utc) if start_time_ms > 0 else None
        end_time = datetime.fromtimestamp(end_time_ms / 1000.0, tz=timezone.utc) if end_time_ms > 0 else None
        
        duration_seconds = (end_time_ms - start_time_ms) / 1000.0 if (end_time_ms > 0 and start_time_ms > 0) else 0.0

        # Parse individual tasks
        tasks = []
        for task in run.get("tasks", []):
            task_state = task.get("state", {})
            task_lc_state = task_state.get("life_cycle_state", "UNKNOWN")
            task_result_state = task_state.get("result_state")  # Can be None for PENDING tasks
            
            # Special handling for unexecuted/pending tasks after predecessor failure
            if task_lc_state == "PENDING" and task_result_state is None:
                task_status = "PENDING"
            elif task_result_state:
                task_status = task_result_state
            else:
                task_status = task_lc_state

            tasks.append({
                "task_key": task.get("task_key"),
                "run_id": str(task.get("run_id")),
                "status": task_status,
                "state_message": task_state.get("state_message", "") or "",
                "attempt_number": task.get("attempt_number", 0)
            })

        return RunRecord(
            run_id=run_id,
            job_id=str(job_id) if job_id else "",
            job_name=job_name,
            status=status,
            state_message=state_message,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration_seconds,
            tasks=tasks
        )