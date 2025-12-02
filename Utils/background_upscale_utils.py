import argparse
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import contextlib
import cv2
from multiprocessing import cpu_count
import numpy as np
import onnxruntime as ort
from PIL import Image
from tqdm import tqdm
import os
import cv2
import threading
import queue
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from Utils.main_utils import slice_video, get_video_duration, concat_videos, repair_video_file
import sys

@contextlib.contextmanager
def load_onnx_session(model_path: str, use_gpu: bool = True):
    providers = ["CUDAExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
    sess_options = ort.SessionOptions()
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.inter_op_num_threads = 0
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.add_session_config_entry("session.intra_op.allow_spinning", "1")
    session = ort.InferenceSession(
        model_path, providers=providers, options=sess_options
    )
    if session is None:
        raise RuntimeError(f"Failed to load ONNX model: {model_path}")
    try:
        yield session
    finally:
        session._sess = (
            None  # Explicitly release resources (optional, for some ORT versions)
        )
        del session


def create_feather_mask(tile_size: int = 256, fade: int = 10) -> np.ndarray:
    m = np.ones((tile_size, tile_size), np.float32)
    ramp = np.linspace(0, 1, fade)
    m[:fade, :] *= ramp[:, None]
    m[-fade:, :] *= ramp[::-1][:, None]
    m[:, :fade] *= ramp[None, :]
    m[:, -fade:] *= ramp[::-1][None, :]
    return m


# These globals will be initialized once per worker
ort_session = None
FEATHER = None
_TILE_SIZE = None
_OVERLAP = None
_SCALE = None


def _upscale_frame_np(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Upscale a single BGR frame (H×W×3 uint8) via tiled ONNX.
    Returns an uint8 BGR frame at (_SCALE*H × _SCALE*W).
    """
    # to 0–1 float
    img = frame_bgr[:, :, ::-1].astype(np.float32) / 255.0  # BGR→RGB
    H, W, _ = img.shape
    out_H, out_W = H * _SCALE, W * _SCALE

    out = np.zeros((out_H, out_W, 3), np.float32)
    wgt = np.zeros((out_H, out_W, 1), np.float32)
    stride = _TILE_SIZE - 2 * _OVERLAP
    inp_name = ort_session.get_inputs()[0].name

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y2 = min(y + _TILE_SIZE, H)
            x2 = min(x + _TILE_SIZE, W)
            th, tw = y2 - y, x2 - x

            tile = img[y:y2, x:x2]
            # resize to model input
            tr = cv2.resize(
                tile, (_TILE_SIZE, _TILE_SIZE), interpolation=cv2.INTER_LINEAR
            )
            data = tr.transpose(2, 0, 1)[None].astype(np.float32)

            # run ONNX
            pred = ort_session.run(None, {inp_name: data})[0][0].transpose(1, 2, 0)
            pred = np.clip(pred, 0, 1)

            # back to tile size
            pr = cv2.resize(
                pred, (tw * _SCALE, th * _SCALE), interpolation=cv2.INTER_CUBIC
            )
            mask = FEATHER[:th, :tw]
            mask = cv2.resize(
                mask, (tw * _SCALE, th * _SCALE), interpolation=cv2.INTER_LINEAR
            )[..., None]

            oy, ox = y * _SCALE, x * _SCALE
            out[oy : oy + th * _SCALE, ox : ox + tw * _SCALE] += pr * mask
            wgt[oy : oy + th * _SCALE, ox : ox + tw * _SCALE] += mask

    wgt[wgt == 0] = 1.0
    final = (out / wgt).clip(0, 1)
    # back to 0–255 uint8 BGR
    final = (final * 255).astype(np.uint8)
    #return cv2.resize(final[:, :, ::-1], (W, H))  # RGB→BGR
    return final[:, :, ::-1]


import os
import cv2
from pathlib import Path
from queue import PriorityQueue
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tqdm import tqdm
    #from stqdm import stqdm as tqdm
except ImportError:
    from tqdm import tqdm


import os
import queue
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import cv2
import torch

@torch.no_grad()
def upscale_video_onnx(
    input_vid: str,
    output_vid: str,
    model_path: str,
    tile_size: int = 128,
    overlap: int = 10,
    scale: float = 4,
    use_gpu: bool = True,
    num_workers: Optional[int] = 8,
    progress_bar=None,  # <-- a Streamlit DeltaGenerator (container) or None
):
    # prepare output folder
    Path("restored").mkdir(exist_ok=True)

    # open once to grab metadata
    cap = cv2.VideoCapture(input_vid)
    if not cap.isOpened():
        raise IOError(f"Cannot open {input_vid}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("No frames read!")
    H, W = frame.shape[:2]
    out_size = (W, H)
    out_size = (W * scale, H * scale)
    out_size = (int(W * scale), int(H * scale))

    cap.release()

    # optional Streamlit progress bar
    if progress_bar is not None:
        # render an actual progress bar inside the container
        pb = progress_bar.progress(0)
    else:
        pb = None

    # init ONNX session & globals
    with load_onnx_session(model_path, use_gpu=use_gpu) as session:
        global ort_session, FEATHER, _TILE_SIZE, _OVERLAP, _SCALE
        ort_session = session
        _TILE_SIZE = tile_size
        _OVERLAP = overlap
        _SCALE = scale
        FEATHER = create_feather_mask(tile_size, fade=overlap)

        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_vid, fourcc, fps, out_size)

        # threading primitives
        frame_queue = queue.PriorityQueue()
        condition = threading.Condition()
        from tqdm import tqdm
        # Writer thread: pulls frames in order and updates the bar
        def writer_fn():
            next_idx = 0
            bar = tqdm(total=total, desc="Writing")
            while next_idx < total:
                with condition:
                    # wait until the next frame is ready
                    while frame_queue.empty() or frame_queue.queue[0][0] != next_idx:
                        condition.wait()
                    _, up = frame_queue.get()
                    writer.write(up)
                    next_idx += 1
                    bar.update(1)
                    # update our progress bar
                    if pb is not None:
                        percent = int((next_idx / total) * 100)
                        pb.progress(percent)

                    condition.notify_all()
            bar.close()
        writer_thread = threading.Thread(target=writer_fn, daemon=True)
        writer_thread.start()

        # Worker: upscale & dispatch to the priority queue
        def worker(frame, idx):
            up = _upscale_frame_np(frame)
            with condition:
                frame_queue.put((idx, up))
                condition.notify_all()

        # Submit all frames
        cap = cv2.VideoCapture(input_vid)
        max_workers = num_workers or os.cpu_count()
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            print(
                f"[INFO] Upscaling {total} frames with up to {max_workers} in flight…,  scale is {scale}"
            )
            for idx in tqdm(range(total), desc="Submitting"):
                ret, frm = cap.read()
                if not ret:
                    break
                exe.submit(worker, frm, idx)

        # Wait for writer to finish
        writer_thread.join()
        cap.release()
        writer.release()

    # Make sure bar hits 100%
    if pb is not None:
        pb.progress(100)

    # Repair and finish
    repair_video_file(str(output_vid))
    print(f"Done → {output_vid}")
    del ort_session, FEATHER, _TILE_SIZE, _OVERLAP, _SCALE
    del session
    import gc
    gc.collect()
    return output_vid


    

def background_upscale_video_onnx_cached(
    input_path,
    output_path,
    scale,
    model_path='models/Real-ESRGAN-General-x4v3.onnx',
    tile_size=128,
    overlap=16,
    use_gpu=True,
    workers=4
):
    """
    Runs ONNX-based video background upscaling with caching.

    Parameters:
    - input_path: Path to the input video (e.g., restored.mp4)
    - output_path: Path to save the upscaled output
    - scale: Upscaling factor
    - model_path: Path to the ONNX model file
    - tile_size: Size of each tile used in upscaling
    - overlap: Overlap between tiles
    - use_gpu: Whether to use GPU
    - workers: Number of parallel workers
    """
    if os.path.exists(output_path):
        print(f"[CACHE] Background upscaled video found: {output_path}")
        return output_path

    print("[INFO] Running ONNX background upscaling...")
    from Utils.background_upscale_utils import upscale_video_onnx

    try:
        upscale_video_onnx(
            input_path,
            output_path,
            model_path=model_path,
            tile_size=tile_size,
            overlap=overlap,
            scale=scale,
            use_gpu=use_gpu,
            num_workers=workers,
        )
    except (Exception, KeyboardInterrupt) as e:
        print(f"[ERROR] ONNX background upscaling interrupted: {e}")
        if os.path.exists(output_path):
            print(f"[CLEANUP] Removing partial output video: {output_path}")
            os.remove(output_path)
        raise

    print(f"[DONE] Background upscaled video created at: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Slice a long video into chunks, upscale in parallel, then concatenate."
    )
    parser.add_argument("input", help="Path to input video file")
    parser.add_argument("output", help="Path to write the upscaled video")
    parser.add_argument("--chunks", type=int, default=4, help="(Unused here) Number of slices")
    parser.add_argument(
        "--model-path",
        default="models/Real-ESRGAN-General-x4v3.onnx",
        help="Path to the ONNX model file",
    )
    parser.add_argument("--tile-size", type=int, default=128, help="Tile size for upscaling")
    parser.add_argument("--overlap", type=int, default=16, help="Overlap size for tiles")
    parser.add_argument("--scale", type=float, default=4, help="Upscaling factor")
    parser.add_argument(
        "--no-gpu",
        dest="use_gpu",
        action="store_false",
        help="Disable GPU and use CPU only",
    )
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    args = parser.parse_args()

    background_upscale_video_onnx_cached(
        args.input,
        args.output,
        scale=args.scale,
        model_path=args.model_path,
        tile_size=args.tile_size,
        overlap=args.overlap,
        use_gpu=args.use_gpu,
        workers=args.workers
    )

