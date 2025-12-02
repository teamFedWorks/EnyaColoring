import uuid
import subprocess
import threading
from typing import Dict, Any, Callable

JOBS: Dict[str, Dict[str, Any]] = {}
LISTENERS: Dict[str, Callable] = {}

def notify(job_id, data):
    if job_id in LISTENERS:
        LISTENERS[job_id](data)

def run_pipeline(job_id: str, node_ip: str, video_path: str, flags: list):
    JOBS[job_id]["status"] = "running"
    notify(job_id, {"status": "running", "progress": 1})

    cmd = ["docker", "exec", "-i", node_ip, "python", "pipeline.py", video_path] + flags
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    for line in proc.stdout:
        if "frame=" in line and "fps" in line:
            try:
                frame = int(line.split("frame=")[1].split()[0])
                JOBS[job_id]["progress"] = min(100, int((frame / 1000) * 100))
                notify(job_id, {"progress": JOBS[job_id]["progress"]})
            except:
                pass

    proc.wait()

    JOBS[job_id]["status"] = "completed"
    JOBS[job_id]["progress"] = 100
    notify(job_id, {"status": "completed", "progress": 100})
