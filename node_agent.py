# /workspace/node_agent.py
import asyncio, json, os, re, shutil, subprocess, uuid
from pathlib import Path
from typing import Dict, List, Optional
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import subprocess, json
import time
import GPUtil
import psutil

app = FastAPI(title="Node-Agent", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)



INPUT_DIR = Path("/workspace/input_videos").resolve()
OUTPUT_DIR = Path("/workspace/output_videos").resolve()
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class Job:
    def __init__(self, job_id: str, input_name: str):
        self.id = job_id
        self.input_name = input_name
        self.status = "queued"   # queued|running|completed|failed
        self.progress = 0
        self.output_path: Optional[str] = None
        self.proc: Optional[asyncio.subprocess.Process] = None
        self.clients: List[WebSocket] = []

    async def broadcast(self, payload: dict):
        dead = []
        for ws in self.clients:
            try: await ws.send_text(json.dumps(payload))
            except Exception: dead.append(ws)
        for ws in dead:
            try: self.clients.remove(ws)
            except ValueError: pass

JOBS: Dict[str, Job] = {}

GPU_CACHE = {"time": 0, "data": None}

def get_gpu_info():
    global GPU_CACHE
    now = time.time()
    if GPU_CACHE["data"] and now - GPU_CACHE["time"] < 3:
        return GPU_CACHE["data"]

    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,temperature.gpu,utilization.gpu",
             "--format=csv,noheader,nounits"]
        ).decode().strip().split("\n")

        detail = []
        for line in raw:
            name, mem_total, mem_used, temp, util = [x.strip() for x in line.split(",")]
            detail.append({
                "name": name,
                "memory_total": int(mem_total),
                "memory_used": int(mem_used),
                "temperature": int(temp),
                "utilization": int(util)
            })

        result = {
            "gpus": len(detail),
            "idle_gpus": sum(1 for d in detail if d["utilization"] < 10),
            "detail": detail,
            "running_jobs": {
                    j_id: {"progress": j.progress, "status": j.status}
                    for j_id, j in JOBS.items()
                    if j.status == "running" or j.status == "queued"
                }
        }

        GPU_CACHE = {"time": now, "data": result}
        return result
    except:
        return {"gpus": 0, "idle_gpus": 0, "detail": [], "running_jobs": {}}

        
def get_gpu_status():
    try:
        if not torch.cuda.is_available():
            return {"gpus": 0, "idle_gpus": 0, "detail": []}

        count = torch.cuda.device_count()
        detail = []
        idle = 0

        for i in range(count):
            total = torch.cuda.get_device_properties(i).total_memory / (1024**2)
            used = torch.cuda.memory_allocated(i) / (1024**2)
            util = (used / total) * 100 if total > 0 else 0
            detail.append({
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "util": round(util, 1),
                "mem_used": round(used, 1),
                "mem_total": round(total, 1)
            })
            if util < 20:
                idle += 1

        return {"gpus": count, "idle_gpus": idle, "detail": detail}

    except:
        return {"gpus": 0, "idle_gpus": 0, "detail": []}


@app.get("/health")
async def health():
    return {"ok": True, "cwd": str(Path.cwd())}


# @app.get("/status")
# async def status():
#     try:
#         result = subprocess.check_output(
#             [
#                 "nvidia-smi",
#                 "--query-gpu=name,memory.total,memory.used,temperature.gpu,utilization.gpu",
#                 "--format=csv,noheader,nounits"
#             ],
#             stderr=subprocess.STDOUT,
#         ).decode().strip().splitlines()

#         gpus = []
#         idle = 0

#         for row in result:
#             name, mem_total, mem_used, temp, util = [x.strip() for x in row.split(",")]
#             util = int(util)
#             mem_used = int(mem_used)
#             mem_total = int(mem_total)

#             if util < 10 and mem_used < (mem_total * 0.15):
#                 idle += 1

#             gpus.append({
#                 "name": name,
#                 "memory_total": mem_total,
#                 "memory_used": mem_used,
#                 "temperature": int(temp),
#                 "utilization": util
#             })

#         return {
#             "gpus": len(gpus),
#             "idle_gpus": idle,
#             "detail": gpus,
#             "running_jobs": {j.id: {"progress": j.progress, "status": j.status} for j in JOBS.values() if j.status == "running"}
#         }

#     except Exception as e:
#         return {"error": str(e), "gpus": 0, "idle_gpus": 0, "detail": [], "running_jobs": {}}


@app.get("/status")
async def status():
    """
    Returns GPU usage and running jobs info so gateway dashboard can show live state.
    """
    try:
        gpus = GPUtil.getGPUs()
        detail = []
        idle = 0

        for g in gpus:
            mem_total = g.memoryTotal
            mem_used = g.memoryUsed
            temp = g.temperature
            util = g.load * 100

            detail.append({
                "name": g.name,
                "memory_total": int(mem_total),
                "memory_used": int(mem_used),
                "temperature": int(temp),
                "utilization": int(util),
            })

            if util < 10 and mem_used < (0.2 * mem_total):
                idle += 1

        return {
            "gpus": len(gpus),
            "idle_gpus": idle,
            "detail": detail,
            "running_jobs": {
                j.id: {"status": j.status, "progress": j.progress}
                for j in JOBS.values()
            },
        }

    except Exception as e:
        return {"error": str(e)}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    target = INPUT_DIR / file.filename
    with target.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"saved": str(target)}

@app.post("/run")
async def run(
    input_name: str = Form(...),
    unet: str = Form("true"),
    face_restore: str = Form("False"),
    upscale: str = Form("False"),
    upscale_value: str = Form("2.0"),
    clahe: str = Form("False"),
):
    src = INPUT_DIR / Path(input_name).name
    if not src.exists():
        return JSONResponse({"error": f"Input not found: {src}"}, status_code=400)

    job_id = uuid.uuid4().hex[:12]
    job = Job(job_id, src.name)
    JOBS[job_id] = job

    args = [
        "python", "/workspace/pipeline.py", str(src),
        str(unet), str(face_restore), str(upscale), str(upscale_value), str(clahe)
    ]

    pct_re = re.compile(r"(\d{1,3})%")
    def maybe_pct(line: str) -> Optional[int]:
        m = pct_re.search(line)
        if m:
            try:
                v = int(m.group(1))
                return max(0, min(100, v))
            except: pass
        return None

    def maybe_output(line: str) -> Optional[str]:
        for key in ("final postprocessed video at:", "final video at:"):
            if key in line:
                return line.split(key, 1)[1].strip()
        return None

    async def reader(stream):
        while True:
            line = await stream.readline()
            if not line: break
            text = line.decode(errors="ignore").rstrip()
            await job.broadcast({"type": "log", "line": text})
            p = maybe_pct(text)
            if p is not None:
                job.progress = p
                await job.broadcast({"type": "progress", "progress": p})
            outp = maybe_output(text)
            if outp: job.output_path = str(Path(outp).resolve())

    async def run_task():
        job.status = "running"
        await job.broadcast({"type": "status", "status": job.status})
        job.proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
        )
        await reader(job.proc.stdout)
        rc = await job.proc.wait()
        job.status = "completed" if rc == 0 else "failed"
        if rc == 0: job.progress = 100
        await job.broadcast({"type": "status", "status": job.status, "progress": job.progress, "output": job.output_path})

    asyncio.create_task(run_task())
    return {"id": job_id}

@app.get("/jobs/{job_id}")
async def job_info(job_id: str):
    j = JOBS.get(job_id)
    if not j: return JSONResponse({"error": "not found"}, status_code=404)
    return {"id": j.id, "input": j.input_name, "status": j.status, "progress": j.progress, "output": j.output_path}

@app.websocket("/ws/jobs/{job_id}")
async def ws_job(ws: WebSocket, job_id: str):
    await ws.accept()
    j = JOBS.get(job_id)
    if not j:
        await ws.send_text(json.dumps({"type":"error","message":"job not found"}))
        return await ws.close()
    j.clients.append(ws)
    try:
        await ws.send_text(json.dumps({"type":"hello","status": j.status, "progress": j.progress}))
        while True:
            await ws.receive_text()  # keepalive (client can send ping)
    except WebSocketDisconnect:
        pass
    finally:
        try: j.clients.remove(ws)
        except ValueError: pass

@app.get("/jobs/{job_id}/output")
async def download(job_id: str):
    j = JOBS.get(job_id)
    if not j or j.status != "completed" or not j.output_path:
        return JSONResponse({"error": "not ready"}, status_code=400)
    path = Path(j.output_path)
    if not path.exists():
        return JSONResponse({"error": "file missing"}, status_code=404)
    return FileResponse(path, filename=path.name, media_type="video/mp4")


@app.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    j = JOBS.get(job_id)
    if not j:
        return JSONResponse({"error": "not found"}, status_code=404)

    if j.proc and j.status == "running":
        j.proc.terminate()
        j.status = "failed"
        await j.broadcast({"type": "status", "status": j.status, "message": "Job cancelled"})
        return {"ok": True, "status": "cancelled"}

    return {"ok": False, "message": "Job is not running"}

