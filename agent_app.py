# agent_app.py
import os
import uuid
import shutil
import asyncio
import orjson
import subprocess
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

INPUT_DIR = Path("/workspace/input_videos")
OUTPUT_DIR = Path("/workspace/output_videos")

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="GPU Agent", version="1.0")

# Allow your portal origin only (adjust in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later to your portal origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job registry (replace with Redis/db later if needed)
class Job:
    def __init__(self, job_id: str, infile: Path):
        self.id = job_id
        self.infile = infile
        self.status = "queued"
        self.progress = 0
        self.stage = "init"
        self.output_path: Optional[Path] = None
        self.proc: Optional[subprocess.Popen] = None
        self.ws_clients: set[WebSocket] = set()

jobs: Dict[str, Job] = {}

def j(data: dict) -> bytes:
    # compact, fast JSON
    return orjson.dumps(data)

async def broadcast(job: Job, payload: dict):
    if not job.ws_clients:
        return
    dead = []
    text = j(payload).decode("utf-8")
    for ws in list(job.ws_clients):
        try:
            await ws.send_text(text)
        except WebSocketDisconnect:
            dead.append(ws)
        except Exception:
            dead.append(ws)
    for ws in dead:
        job.ws_clients.discard(ws)

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    # Save to /workspace/input_videos
    fname = f"{uuid.uuid4().hex}_{file.filename}"
    dest = INPUT_DIR / fname
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"ok": True, "path": str(dest)}

@app.post("/api/jobs")
async def start_job(
    input_path: str = Form(...),
    unet: str = Form(...),
    face_restore: str = Form(...),
    upscale: str = Form(...),
    upscale_value: str = Form("2.0"),
    clahe: str = Form("false"),
):
    infile = Path(input_path)
    if not infile.exists():
        return JSONResponse({"error": "input not found"}, status_code=400)

    job_id = uuid.uuid4().hex[:12]
    job = Job(job_id, infile)
    jobs[job_id] = job

    # Build argv for your pipeline.py (exactly like you run it from the UI)
    argv = [
        "python", "pipeline.py",
        str(infile),
        unet, face_restore, upscale, upscale_value, clahe
    ]

    # Launch subprocess, stream stdout lines into WS as telemetry
    job.proc = subprocess.Popen(
        argv,
        cwd="/workspace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True
    )
    job.status = "running"
    job.stage  = "launched"

    async def reader():
        # Very light parser: try to infer progress & stage
        try:
            assert job.proc and job.proc.stdout
            async def send(stage=None, progress=None, level="info", msg=None):
                if stage: job.stage = stage
                if progress is not None:
                    job.progress = max(0, min(100, int(progress)))
                payload = {
                    "type": "telemetry",
                    "job_id": job.id,
                    "status": job.status,
                    "stage": job.stage,
                    "progress": job.progress,
                }
                if msg:
                    payload["log"] = {"level": level, "message": msg}
                await broadcast(job, payload)

            await send(stage="starting", progress=0, msg="Job started")

            # Read lines and heuristically map to stage/progress
            for line in job.proc.stdout:
                line = line.rstrip("\n")
                # stage hints
                if "Running restoration" in line or "Restoring:" in line:
                    await send(stage="restore", msg=line)
                elif "Starting scene detection" in line or "Scene split" in line:
                    await send(stage="scene_split", msg=line)
                elif "ComfyUI" in line:
                    await send(stage="comfyui", msg=line)
                elif "colorized" in line.lower():
                    await send(stage="colorize", msg=line)
                elif "postprocessed" in line.lower():
                    await send(stage="postprocess", msg=line)
                elif "final video at:" in line.lower():
                    # Capture output path
                    outp = line.split("final video at:")[-1].strip()
                    p = Path(outp)
                    if p.exists():
                        job.output_path = p
                    await send(stage="finalize", progress=95, msg=line)
                else:
                    # progress pattern from tqdm like "xx%|"
                    if "%|" in line:
                        # Try to extract the first percentage number
                        try:
                            pct = int(line.split("%|")[0].split()[-1])
                            await send(progress=pct, msg=line)
                        except Exception:
                            await send(msg=line)
                    else:
                        await send(msg=line)

            rc = job.proc.wait()
            if rc == 0:
                job.status = "completed"
                job.progress = 100
                await broadcast(job, {
                    "type": "telemetry",
                    "job_id": job.id,
                    "status": job.status,
                    "stage": job.stage,
                    "progress": job.progress,
                    "log": {"level":"info","message":"Job completed"}
                })
            else:
                job.status = "failed"
                await broadcast(job, {
                    "type": "telemetry",
                    "job_id": job.id,
                    "status": job.status,
                    "stage": job.stage,
                    "progress": job.progress,
                    "log": {"level":"error","message":f"Process exited {rc}"}
                })
        except Exception as e:
            job.status = "failed"
            await broadcast(job, {
                "type":"telemetry","job_id":job.id,"status":job.status,"stage":job.stage,
                "progress": job.progress, "log":{"level":"error","message":repr(e)}
            })

    asyncio.create_task(reader())
    return {"id": job_id, "status": job.status}

@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"error":"not found"}, status_code=404)
    return {
        "id": job.id,
        "status": job.status,
        "progress": job.progress,
        "stage": job.stage,
        "input": str(job.infile),
        "output": str(job.output_path) if job.output_path else None
    }

@app.websocket("/ws/jobs/{job_id}")
async def ws_job(websocket: WebSocket, job_id: str):
    await websocket.accept()
    job = jobs.get(job_id)
    if not job:
        await websocket.send_text(j({"type":"error","message":"job not found"}).decode())
        await websocket.close()
        return
    job.ws_clients.add(websocket)
    # Send snapshot
    await websocket.send_text(j({
        "type": "telemetry", "job_id": job.id, "status": job.status,
        "stage": job.stage, "progress": job.progress
    }).decode())
    try:
        while True:
            # keepalive (no client->server messages expected)
            await websocket.receive_text()
    except WebSocketDisconnect:
        job.ws_clients.discard(websocket)

@app.get("/api/jobs/{job_id}/output")
def get_output(job_id: str):
    job = jobs.get(job_id)
    if not job or job.status != "completed" or not job.output_path or not job.output_path.exists():
        return JSONResponse({"error":"output not ready"}, status_code=404)
    return FileResponse(path=str(job.output_path), filename=job.output_path.name, media_type="video/mp4")

@app.get("/api/health")
def health():
    return {"ok": True}
