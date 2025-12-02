# jobs_state.py
from typing import Dict, Any
from datetime import datetime

JOBS: Dict[str, Dict[str, Any]] = {}
LISTENERS: Dict[str, callable] = {}

def create_job(node_ip, filename):
    job_id = f"job_{len(JOBS)+1:05d}"
    JOBS[job_id] = {
        "id": job_id,
        "node": node_ip,
        "status": "queued",
        "progress": 0,
        "input": filename,
        "output": None,
        "created_at": datetime.utcnow().isoformat()
    }
    return job_id
