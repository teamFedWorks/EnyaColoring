from deoldify import device
from deoldify.device_id import DeviceId
device.set(device=DeviceId.GPU0)

import io
import os
import threading
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
import cv2
import numpy as np
import torch
from PIL import Image
from Utils.main_utils import repair_video_file

# deoldify imports
from deoldify.visualize import get_image_colorizer

# thread‐local storage for colorizer
_thread_locals = threading.local()

def _get_colorizer():
    """Return a per-thread DeOldify colorizer instance."""
    c = getattr(_thread_locals, "colorizer", None)
    if c is None:
        device.set(device=DeviceId.GPU0)
        c = get_image_colorizer(artistic=True)
        _thread_locals.colorizer = c
    return c

def _deoldify_frame(args):
    """
    Worker to colorize a single BGR frame.
    args = (frame_bgr: np.ndarray, render_factor: int)
    Returns colorized BGR frame.
    """
    frame_bgr, render_factor = args
    # BGR → RGB → PIL → PNG bytes
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(frame_rgb)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)

    # colorize
    colorizer = _get_colorizer()
    out_pil = colorizer.get_transformed_image_(
        buf,
        render_factor=render_factor,
        post_process=True,
        watermarked=False
    )
    out_rgb = np.array(out_pil)
    # RGB → BGR
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

def deoldify_parallel(
    input_path: str,
    output_path: str,
    render_factor: int = 16,
    num_workers: Optional[int] = 1,
    max_frames: Optional[int] = None,
):
    """
    Parallel DeOldify colorization of a video.
    - input_path: source video
    - output_path: destination mp4
    - render_factor: DeOldify render factor
    - num_workers: threads to use (defaults to os.cpu_count())
    - max_frames: stop after this many frames
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open {input_path}")

    # metadata
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is not None:
        total = min(total, max_frames)

    # output writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {output_path}")

    max_workers = num_workers or os.cpu_count()
    pending: dict = {}
    results: dict[int, np.ndarray] = {}
    next_to_write = 0

    try:
        #from stqdm import stqdm as tqdm
        from tqdm import tqdm
    except ImportError:
        from tqdm import tqdm

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        print(f"[INFO] Colorizing {total} frames with up to {max_workers} workers…")
        for idx in tqdm(range(total), desc="Submitting frames"):
            ret, frame = cap.read()
            if not ret:
                break

            # throttle: if at capacity, write one result
            if len(pending) >= max_workers:
                done_fut = next(as_completed(pending))
                done_idx = pending.pop(done_fut)
                results[done_idx] = done_fut.result()
                # write any in-order frames
                while next_to_write in results:
                    writer.write(results.pop(next_to_write))
                    next_to_write += 1

            # submit task
            fut = executor.submit(_deoldify_frame, (frame.copy(), render_factor))
            pending[fut] = idx

        # drain remaining
        for done_fut in as_completed(pending):
            done_idx = pending[done_fut]
            results[done_idx] = done_fut.result()

        # write rest in order
        while next_to_write in results:
            writer.write(results.pop(next_to_write))
            next_to_write += 1

    cap.release()
    writer.release()

    # cleanup GPU memory
    try:
        del _thread_locals.colorizer
    except Exception:
        pass
    torch.cuda.empty_cache()

    # optional faststart repair
    repair_video_file(str(output_path))

    print(f"[INFO] Done → {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Parallel DeOldify video colorization")
    parser.add_argument("input",  help="Input video path")
    parser.add_argument("output", help="Output video path")
    parser.add_argument(
        "--render-factor", type=int, default=16,
        help="DeOldify render factor"
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel threads"
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Limit number of frames processed"
    )
    args = parser.parse_args()

    deoldify_parallel(
        input_path     = args.input,
        output_path    = args.output,
        render_factor  = args.render_factor,
        num_workers    = args.workers,
        max_frames     = None,
    )
