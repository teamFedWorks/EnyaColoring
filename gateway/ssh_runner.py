# ssh_runner.py
import paramiko
import re
from jobs_state import JOBS, LISTENERS

REMOTE_GPU_SERVER_IP = "192.168.27.14"
REMOTE_GPU_SERVER_PORT = 8870  # your docker exposed port
REMOTE_USER = "root"           # adjust if needed
REMOTE_PASS = ""               # if password auth, otherwise use key

def run_pipeline(job_id, remote_input_path):
    JOBS[job_id]["status"] = "running"
    
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(REMOTE_GPU_SERVER_IP, username=REMOTE_USER, password=REMOTE_PASS)

    cmd = f"docker exec pipeline_v1 python3 /workspace/pipeline.py {remote_input_path} true false false 1.0 false"
    stdin, stdout, stderr = ssh.exec_command(cmd)

    progress_re = re.compile(r"(\d+)%\|")

    for line in iter(stdout.readline, ""):
        line = line.strip()

        match = progress_re.search(line)
        if match:
            pct = int(match.group(1))
            JOBS[job_id]["progress"] = pct
            
            if job_id in LISTENERS:
                LISTENERS[job_id](f'{{"progress": {pct}}}')

    JOBS[job_id]["status"] = "completed"
