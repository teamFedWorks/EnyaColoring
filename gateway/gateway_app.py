# # gateway_app.py
# from fastapi import FastAPI, UploadFile, Form
# from fastapi.responses import FileResponse
# from sse_starlette.sse import EventSourceResponse
# import shutil
# import threading
# import os

# from jobs_state import JOBS, LISTENERS, create_job
# from ssh_runner import run_pipeline

# app = FastAPI()

# LOCAL_INPUT_DIR = "/workspace/input_videos/"
# LOCAL_OUTPUT_DIR = "/workspace/output_videos/"

# os.makedirs(LOCAL_INPUT_DIR, exist_ok=True)
# os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)

# @app.post("/jobs")
# async def start_job(
#     node_ip: str = Form(...),
#     unet: bool = Form(...),
#     face_restore: bool = Form(...),
#     upscale: bool = Form(...),
#     upscale_value: float = Form(...),
#     file: UploadFile = None
# ):
#     filename = file.filename
#     local_path = os.path.join(LOCAL_INPUT_DIR, filename)

#     with open(local_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)

#     job_id = create_job(node_ip, filename)

#     thread = threading.Thread(target=run_pipeline, args=(job_id, local_path))
#     thread.start()

#     return JOBS[job_id]

# @app.get("/jobs/{job_id}")
# async def get_job(job_id: str):
#     return JOBS.get(job_id, {})

# @app.get("/jobs/{job_id}/events")
# def stream_job(job_id: str):
#     def streamer(callback):
#         LISTENERS[job_id] = callback
#         while True:
#             pass

#     return EventSourceResponse(streamer)

# @app.get("/jobs/{job_id}/output")
# async def download(job_id: str):
#     output_path = JOBS[job_id]["output"]
#     return FileResponse(output_path, filename=os.path.basename(output_path))





# gateway_app.py — FastAPI gateway (no SSE)
import io, os, httpx, json
from fastapi import FastAPI, UploadFile, File, Form, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

app = FastAPI(title="Video Gateway", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# --- Configure ALL your nodes here (ip:port). You said MULTIPLE ---
NODES = [
    {"id": "gpu-a", "name": "GPU A", "base": "http://192.168.27.14:9090"},  # example node-agent port
    # add more like:
    # {"id": "gpu-b", "name": "GPU B", "base": "http://192.168.1.6:9090"},
]

# If you really want to use 8870, the node-agent itself must listen there.
# Otherwise keep 9090 for the agent and map it in docker: 9090:9090

@app.get("/nodes")
async def nodes():
    # Return ip:port so the UI can target the exact agent
    rows = []
    for n in NODES:
        rows.append({
            "id": n["id"], "name": n.get("name"),
            "ip": n["base"].replace("http://",""),   # e.g. 192.168.27.14:9090
            "gpus": None, "idle_gpus": None, "status": "online",
        })
    return rows

@app.post("/jobs")
async def create_job(
    node_ip: str = Form(...),
    unet: bool = Form(True),
    face_restore: bool = Form(False),
    upscale: bool = Form(False),
    upscale_value: float = Form(2.0),
    file: UploadFile = File(...),
):
    base = f"http://{node_ip}"
    async with httpx.AsyncClient(timeout=120) as client:
        # upload
        mp = httpx.MultipartWriter()
        mp.add_file("file", await file.read(), filename=file.filename, content_type=file.content_type)
        r = await client.post(f"{base}/upload", content=mp, headers={"Content-Type": mp.content_type})
        r.raise_for_status()
        saved = r.json()["saved"]

    async with httpx.AsyncClient(timeout=None) as client:
        # run
        data = {
            "input_name": os.path.basename(saved),
            "unet": str(unet),
            "face_restore": str(face_restore),
            "upscale": str(upscale),
            "upscale_value": str(upscale_value),
            "clahe": "False",
        }
        r2 = await client.post(f"{base}/run", data=data)
        r2.raise_for_status()
        return r2.json()  # { id }

@app.websocket("/ws/jobs/{node_ip}/{job_id}")
async def ws_proxy(ws: WebSocket, node_ip: str, job_id: str):
    await ws.accept()
    target = f"ws://{node_ip}/ws/jobs/{job_id}"
    import websockets
    try:
        async with websockets.connect(target) as remote:
            await ws.send_text(json.dumps({"type": "proxy", "ok": True}))
            async for msg in remote:
                await ws.send_text(msg)
    except Exception as e:
        await ws.send_text(json.dumps({"type":"error","message":str(e)}))
    finally:
        await ws.close()

@app.get("/jobs/{node_ip}/{job_id}")
async def job_info(node_ip: str, job_id: str):
    base = f"http://{node_ip}"
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{base}/jobs/{job_id}")
        return JSONResponse(r.json(), status_code=r.status_code)

@app.get("/jobs/{node_ip}/{job_id}/output")
async def job_output(node_ip: str, job_id: str):
    base = f"http://{node_ip}"
    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.get(f"{base}/jobs/{job_id}/output")
        if r.status_code != 200:
            return JSONResponse(r.json(), status_code=r.status_code)
        return StreamingResponse(
            io.BytesIO(r.content),
            media_type=r.headers.get("content-type","video/mp4"),
            headers={"Content-Disposition": f"attachment; filename=output_{job_id}.mp4"},
        )
