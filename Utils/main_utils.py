import os
import subprocess
import shutil
import hashlib
from pathlib import Path
import cv2
import torch, gc
import time
import unicodedata
import yt_dlp
from tqdm import tqdm
import sys
import tempfile
import numpy as np
import re

#OUTPUT_ROOT = "output_videos_latest"
OUTPUT_ROOT = os.getenv("OUTPUT_ROOT", "output_videos_latest")
INPUT_ROOT = os.getenv("INPUT_ROOT", "input_videos")

def get_output_dir(actual_input_path: str, original_input_path: str, task_name: str) -> Path:
    """
    Create output dir like: output_videos/<original_basename>/<task_name>__<hash(actual_input)>
    
    - actual_input_path: the file used as input (can be intermediate)
    - original_input_path: the root video file (used to group all 8 tasks)
    - task_name: e.g., "restore_bw_film"
    """
    original_base = Path(original_input_path).stem
    abs_actual_input = str(Path(actual_input_path).resolve())
    input_hash = hashlib.sha1(abs_actual_input.encode()).hexdigest()[:12]
    folder_name = f"{task_name}__{input_hash}"
    full_path = Path(OUTPUT_ROOT) / original_base / folder_name
    full_path.mkdir(parents=True, exist_ok=True)
    return full_path

def get_cached_file(actual_input_path: str, step_name: str, video_source_path: str) -> str:
    """
    Returns cached file path using new folder structure.
    - actual_input_path: the input being processed
    - video_source_path: the original base input
    - step_name: logical name of the file, e.g. 'restored'
    """
    output_dir = get_output_dir(actual_input_path, video_source_path, step_name)
    return str(output_dir / f"{step_name}.mp4")


def sanitize_latin_filename(filename):
    """Keep only Latin letters, numbers, and safe symbols. Truncate and add timestamp/hash."""
    filename = os.path.basename(filename)
    filename = unicodedata.normalize("NFKC", filename)
    allowed = []
    for c in filename:
        cat = unicodedata.category(c)
        if (cat.startswith("L") or cat.startswith("N")) and (
            "LATIN" in unicodedata.name(c, "") or c.isdigit()
        ):
            allowed.append(c)
        elif c in ["_", "-", "."]:
            allowed.append(c)
    safe = "".join(allowed)
    safe = safe[:32]
    hash_part = hashlib.sha1(filename.encode("utf-8")).hexdigest()[:8]
    timestamp = time.strftime("%Y%m%d%H%M%S")
    return f"{safe}_{timestamp}_{hash_part}"

def get_input_video_path(youtube_url=None, manual_path=None):
    """
    Download YouTube video if URL is given, or fallback to manual path.
    Returns the final input_video_path.
    """
    Path("input_videos").mkdir(exist_ok=True)

    if youtube_url:
        output_template = "input_videos/%(title)s.%(ext)s"
        ydl_opts = {
            "outtmpl": output_template,
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",
            "merge_output_format": "mp4",
            "quiet": True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=True)
                orig_path = ydl.prepare_filename(info)
                base, ext = os.path.splitext(os.path.basename(orig_path))
                safe_base = sanitize_latin_filename(base)
                safe_path = os.path.join("input_videos", f"{safe_base}{ext}")
                if orig_path != safe_path:
                    os.rename(orig_path, safe_path)
                print(f"[✅] Downloaded and saved to: {safe_path}")
                return safe_path
        except Exception as e:
            print(f"[❌] Download failed: {e}")
            return None

    elif manual_path and os.path.exists(manual_path):
        print(f"[ℹ️] Using manual path: {manual_path}")
        return manual_path

    else:
        raise ValueError("No valid YouTube URL or manual path provided.")


def get_input_video_path_2(youtube_url=None, manual_path=None):
    """
    Download YouTube video (with audio) or use a manual file path.
    Saves to input_videos/ with timestamped and safe filenames.
    Returns the final path.
    """
    Path("input_videos").mkdir(exist_ok=True)

    if youtube_url:
        import yt_dlp  # lazy import

        # Basic download target (temporary, will rename later)
        output_template = "input_videos/%(title)s.%(ext)s"

        ydl_opts = {
            "outtmpl": output_template,
            "format": "bestvideo+bestaudio/best",  # Ensure merged AV
            "merge_output_format": "mp4",
            "postprocessors": [{
                "key": "FFmpegVideoConvertor",
                "preferedformat": "mp4",
            }],
            "quiet": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=True)
                orig_path = ydl.prepare_filename(info)

                # Build new timestamped, sanitized filename
                base, ext = os.path.splitext(os.path.basename(orig_path))
                # safe_base = sanitize_latin_filename(base)
                # timestamp = time.strftime("%Y%m%d_%H%M%S")
                # safe_path = os.path.join("input_videos", f"{safe_base}_{timestamp}.mp4")
                safe_base = sanitize_latin_filename(base)
                safe_path = os.path.join("input_videos", f"{safe_base}{ext}")

                # Rename if needed
                if orig_path != safe_path and os.path.exists(orig_path):
                    os.rename(orig_path, safe_path)

                # Check if audio exists
                result = subprocess.run(
                    ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_type", "-of", "default=noprint_wrappers=1", safe_path],
                    capture_output=True, text=True
                )
                if "codec_type=audio" not in result.stdout:
                    print(f"[❌] Warning: No audio stream found in: {safe_path}")
                else:
                    print(f"[✅] Downloaded with audio: {safe_path}")

                return safe_path

        except Exception as e:
            print(f"[❌] Download failed: {e}")
            return None

    elif manual_path and os.path.exists(manual_path):
        print(f"[ℹ️] Using manual path: {manual_path}")
        return manual_path

    else:
        raise ValueError("No valid YouTube URL or manual path provided.")

def get_input_video_path(youtube_url=None, manual_path=None):
    """
    Uses yt-dlp with strict AVC1 video and M4A audio format selection.
    Applies safe, timestamped filename using sanitize_latin_filename().
    """
    Path(INPUT_ROOT).mkdir(exist_ok=True)

    if youtube_url:
        try:
            # Step 1: Get video info JSON (title etc.)
            cmd_info = [
                "yt-dlp", "--print", "%(title)s.%(ext)s", "--get-filename",
                "-f", "bv*[vcodec^=avc1][ext=mp4]+ba[ext=m4a]/b[ext=mp4]",
                youtube_url
            ]
            raw_output = subprocess.check_output(cmd_info, text=True).strip()
            base = os.path.splitext(os.path.basename(raw_output))[0]
            ext = os.path.splitext(raw_output)[1]

            # Step 2: Sanitize and build safe path
            safe_name = sanitize_latin_filename(base)
            final_path = os.path.join(INPUT_ROOT, f"{safe_name}{ext}")

            # Step 3: Download using yt-dlp with output to sanitized name
            cmd_download = [
                "yt-dlp",
                "-f", "bv*[vcodec^=avc1][ext=mp4]+ba[ext=m4a]/b[ext=mp4]",
                "-o", final_path,
                youtube_url
            ]
            subprocess.run(cmd_download, check=True)
            print(f"[✅] Video downloaded: {final_path}")
            return final_path

        except subprocess.CalledProcessError as e:
            print(f"[❌] yt-dlp download failed: {e}")
            return None

    elif manual_path and os.path.exists(manual_path):
        print(f"[ℹ️] Using manual path: {manual_path}")
        return manual_path

    else:
        raise ValueError("No valid YouTube URL or manual path provided.")

def get_frame_count(input_path):
    """Use ffprobe to get the total number of video frames."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "csv=p=0",
        input_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return int(result.stdout.strip())
    except ValueError:
        return 0


def slice_video(input_vid: str, chunks: int, tmp_dir: Path) -> list[str]:
    """Slice input video into N equal-frame-count chunks using ffmpeg segment muxer."""
    import math
    from main_utils import get_frame_count

    # Get total frames and compute frames per chunk
    total_frames = get_frame_count(input_vid)
    frames_per_chunk = math.ceil(total_frames / chunks)

    # Use ffmpeg segment muxer to split by frame count
    segment_pattern = tmp_dir / "segment_%03d.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-i", input_vid,
        "-c", "copy",
        "-f", "segment",
        "-segment_frames", str(frames_per_chunk),
        "-reset_timestamps", "1",
        str(segment_pattern)
    ]
    subprocess.run(cmd, capture_output=True, check=True)

    # Build list of generated segment paths (up to the requested count)
    segment_paths = []
    for i in range(chunks):
        path = tmp_dir / f"segment_{i:03d}.mp4"
        if not path.exists():
            break
        segment_paths.append(str(path))
    return segment_paths


def get_video_duration(input_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        input_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        raise RuntimeError(f"Could not retrieve duration for {input_path}")



def concat_videos(video_list: list[str], output_vid: str, tmp_dir: Path):
    """Concatenate videos listed in video_list into a single output video."""
    list_file = tmp_dir / "concat_list.txt"
    with open(list_file, "w") as f:
        for path in video_list:
            f.write(f"file '{path}'\n")

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-c",
        "copy",
        output_vid,
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def repair_video_file(output_vid):
    try:
        repaired = output_vid + ".repaired.mp4"
        # cmd = [
        #     "ffmpeg",
        #     "-y",
        #     "-i",
        #     output_vid,
        #     "-movflags",
        #     "+faststart",
        #     repaired,
        # ]
        cmd = [
            "ffmpeg",
            "-y",
            "-i", output_vid,
            "-c:v", "copy",     # Don't re-encode video
            "-c:a", "copy",     # Don't re-encode audio
            "-movflags", "+faststart",
            repaired,
        ]
        subprocess.run(cmd, capture_output=True)
        os.replace(repaired, output_vid)
        print(f"[INFO] ffmpeg repair passthrough complete: {output_vid}")
    except Exception as e:
        print(f"[WARN] ffmpeg repair passthrough failed: {e}")

def restore_bw_film_cached(input_path, first_path):
    """
    Restore B&W film, using filecache to avoid reprocessing.
    - input_path: current input (actual file to process)
    - first_path: root input (used to name main folder)
    """
    output_path = get_cached_file(input_path, "restore_bw_film", video_source_path=first_path)

    if os.path.exists(output_path):
            print(f"[CACHE] Valid restored B&W video found: {output_path}")
            return output_path


    print("[INFO] Running restoration...")
    from Utils.restore_bw_utils import restore_bw_film
    try:
        restore_bw_film(input_path, output_path)
    except (Exception, KeyboardInterrupt) as e:
        print(f"[ERROR] Restoration interrupted: {e}")
        if os.path.exists(output_path):
            print(f"[CLEANUP] Removing partial output video: {output_path}")
            os.remove(output_path)
        raise
    finally:
        import torch, gc
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
    #print(f"[DONE] Restored video created at: {output_path}")
    return output_path


def upscale_faces_cached(input_path, first_path):
    """Upscaling Faces, using filecache to avoid reprocessing."""
    output_path = get_cached_file(input_path, "faces_upscale", video_source_path=first_path)
    if os.path.exists(output_path):
        print(f"[CACHE] Faces upscaled video found: {output_path}")
        return output_path
    print("[INFO] Running Faces Upscaling...")
    from Utils.face_restore_utils import upscale_faces
    try:
        suppress_cpp_stderr()
        upscale_faces(input_path, output_path)
        restore_cpp_stderr()
    except (Exception, KeyboardInterrupt) as e:
        print(f"[ERROR] Faces upscaling interrupted: {e}")
        if os.path.exists(output_path):
            print(f"[CLEANUP] Removing partial output video: {output_path}")
            os.remove(output_path)
        raise
    finally:
        import torch, gc
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
    return output_path



def downscale_video_in_place(input_path, scale=4, min_allowed_dim=256):
    """
    Downscale video by a divisor scale, ensuring the smallest side is at least min_allowed_dim.
    Aspect ratio is preserved if a side drops below the threshold.
    """

    input_cap = cv2.VideoCapture(input_path)
    if not input_cap.isOpened():
        raise ValueError(f"Could not open input video: {input_path}")
    
    input_width = int(input_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_height = int(input_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = input_cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_count = int(input_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initial downscaled resolution
    new_width = int(input_width / scale)
    new_height = int(input_height / scale)

    aspect_ratio = input_width / input_height

    # If either dimension is too small, adjust while preserving aspect ratio
    if new_width < min_allowed_dim or new_height < min_allowed_dim:
        if input_width < input_height:
            new_width = min_allowed_dim
            new_height = int(round(new_width / aspect_ratio))
        else:
            new_height = min_allowed_dim
            new_width = int(round(new_height * aspect_ratio))

    # Skip if resizing is unnecessary
    if new_width == input_width and new_height == input_height:
        print("No resizing needed: target resolution equals input resolution.")
        input_cap.release()
        return

    temp_dir = os.path.dirname(input_path)
    temp_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir, suffix=".mp4")
    temp_path = temp_file.name
    temp_file.close()

    out = cv2.VideoWriter(temp_path, fourcc, fps, (new_width, new_height))
    print(f"Downscaling video by scale : {scale}")
    pbar = tqdm(total=frame_count, desc="📉 Downscaling", unit="frame")
    while True:
        ret, frame = input_cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        out.write(resized)
        pbar.update(1)
    pbar.close()
    input_cap.release()
    out.release()

    shutil.move(temp_path, input_path)
    repair_video_file(str(input_path))
    print(f"✅ Downscaled video saved and replaced original at: {input_path}")


import cv2
import numpy as np
import os
import tempfile
import shutil
from tqdm import tqdm

def apply_clahe_in_place(input_path, output_path=None, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to a grayscale video.
    The result is written to the specified output_path. If no output_path is given, the original file is replaced.
    """
    input_cap = cv2.VideoCapture(input_path)
    if not input_cap.isOpened():
        raise ValueError(f"Could not open input video: {input_path}")

    width = int(input_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(input_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = input_cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(input_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Prepare temporary output file
    if output_path is None:
        temp_dir = os.path.dirname(input_path)
        temp_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir, suffix=".mp4")
        output_path = temp_file.name
        temp_file.close()

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    pbar = tqdm(total=frame_count, desc="✨ Applying CLAHE", unit="frame")
    while True:
        ret, frame = input_cap.read()
        if not ret:
            break

        gray = frame[:, :, 0] if len(frame.shape) == 3 else frame
        enhanced = clahe.apply(gray)
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        out.write(enhanced)
        pbar.update(1)

    pbar.close()
    input_cap.release()
    out.release()

    # Replace original if no output path was specified
    if output_path != input_path:
        shutil.move(output_path, input_path)
        print(f"✅ CLAHE-enhanced video saved and replaced original at: {input_path}")
    else:
        print(f"✅ CLAHE-enhanced video saved to: {output_path}")

    return input_path

# Example usage (not executed here):
# apply_clahe_in_place("path/to/video.mp4")
# apply_clahe_in_place("path/to/video.mp4", output_path="path/to/output.mp4")



    
def background_upscale_video_onnx_cached(input_path, first_path, clahe_flag, scale, model_path='models/Real-ESRGAN-General-x4v3.onnx'):
    """
    Runs ONNX-based video background upscaling with caching.
    
    Parameters:
    - input_path: actual input (e.g. restored.mp4)
    - first_path: original video path 
    - model_path: path to ONNX model file (customizable from calling code)
    """
    output_path = get_cached_file(input_path, "background_upscale", video_source_path=first_path)
    if os.path.exists(output_path):
        print(f"[CACHE] Background upscaled video found: {output_path}")
        return output_path

    print("[INFO] Running ONNX background upscaling...")
    from Utils.background_upscale_utils import upscale_video_onnx
    
    try:
        downscale_video_in_place(input_path, scale)
        if(clahe_flag):
            apply_clahe_in_place(input_path)
        upscale_video_onnx(input_path, output_path, model_path=model_path, scale=scale)
        resize_video_in_place(output_path, first_path)
    except (Exception, KeyboardInterrupt) as e:
        print(f"[ERROR] Onnx background upscaling interrupted: {e}")
        if os.path.exists(output_path):
            print(f"[CLEANUP] Removing partial output video: {output_path}")
            os.remove(output_path)
        raise
    finally:
        import torch, gc
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
    return output_path



import os, pathlib, shutil, cv2, gc
from PIL import Image

# COMFY = "http://192.168.27.12:8188"   # ComfyUI server (default)
# WORKFLOW_JSON = "workflow.json"       # default workflow file


    

def resize_video_in_place(input_path, target_path):
    # Get target resolution
    target_cap = cv2.VideoCapture(target_path)
    if not target_cap.isOpened():
        raise ValueError(f"Could not open target video: {target_path}")
    target_width = int(target_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    target_height = int(target_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_cap.release()

    # Get input resolution
    input_cap = cv2.VideoCapture(input_path)
    if not input_cap.isOpened():
        raise ValueError(f"Could not open input video: {input_path}")
    input_width = int(input_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_height = int(input_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Check if resizing is needed
    if target_width >= input_width and target_height >= input_height:
        print("No resizing needed: Target resolution is not smaller.")
        input_cap.release()
        return  # Exit without resizing

    fps = input_cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(input_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Create a temporary file in the same directory
    temp_dir = os.path.dirname(input_path)
    temp_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir, suffix=".mp4")
    temp_path = temp_file.name
    temp_file.close()

    out = cv2.VideoWriter(temp_path, fourcc, fps, (target_width, target_height))
 
    with tqdm(total=total_frames, desc="🔄 Resizing video", unit="frame") as pbar:
        while True:
            ret, frame = input_cap.read()
            if not ret:
                break
            resized_frame = cv2.resize(frame, (target_width, target_height))
            out.write(resized_frame)
            pbar.update(1)

    input_cap.release()
    out.release()

    # Replace original video with resized one
    shutil.move(temp_path, input_path)
    from Utils.main_utils import repair_video_file
    repair_video_file(str(input_path))
    print(f"Resized video saved and replaced original at {input_path}")


def run_scene_split_cached(input_path: str, original_input_path: str, scale=2) -> str:
    from Utils.pyscene_split_utils import split_video_by_scene, create_video_from_grayscale_frames
    """
    Run scene detection + scene_refs video generation with folder hashing.
    Returns full path to the scene_refs video.
    """
    output_dir = get_output_dir(input_path, original_input_path, "scene_split")
    video_base = Path(input_path).stem
    scene_refs_bw_name = f"{video_base}_bw.mp4"
    image_dir = output_dir / f"{video_base}_images"
    scene_refs_bw_path = os.path.join(image_dir, scene_refs_bw_name)
    if os.path.exists(scene_refs_bw_path):
        print(f"[CACHE] Scene-split video found: {scene_refs_bw_path}")
        return scene_refs_bw_path

    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {input_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
    
        print(f"[INFO] Starting scene detection for: {input_path}")
        split_video_by_scene(input_path, str(output_dir), original_input_path,  scale, fps=fps)
    
        print(f"[INFO] Creating scene_refs video: {scene_refs_bw_path}")
        create_video_from_grayscale_frames(str(image_dir), output_video_name=scene_refs_bw_name, fps=fps)
    except (Exception, KeyboardInterrupt) as e:
        print(f"[ERROR] Scene-split interrupted: {e}")
        if os.path.exists(scene_refs_bw_path):
            print(f"[CLEANUP] Removing partial output video: {scene_refs_bw_path}")
            os.remove(scene_refs_bw_path)
        output_folder =  output_dir 
        if os.path.exists(output_folder) and os.path.isdir(output_folder):
            print(f"[CLEANUP] Removing output folder: {output_folder}")
            shutil.rmtree(output_folder)
        raise
    finally:
        import torch, gc
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f"[DONE] Scene-split refs video saved at: {scene_refs_bw_path}")
    return str(scene_refs_bw_path)





def comfyflux_colorize_video_cached(input_path,
                                    first_path,
                                    prompt_text,
                                    seed=2^24,
                                    steps=10,
                                    cfg=1.0,
                                    flux_guidance=2.5):
    """
    ComfyFlux-based video colorization with caching and resume support.
    - Extracts frames only if no output frames exist.
    - Skips already processed frames on resume.
    - Combines frames into cached .mp4 using input video's FPS and size.
    """

    # --- Cached output path (.mp4) ---
    output_path = get_cached_file(input_path, "comfyflux_colorize", video_source_path=first_path) 
    if os.path.exists(output_path):
        print(f"[CACHE] Found comfyflux video: {output_path}")
        return output_path

    # --- Prepare input/output frames folders ---
    base_dir = pathlib.Path(output_path).parent
    base_name = pathlib.Path(output_path).stem
    frames_in = os.path.join(base_dir, f"{base_name}_frames_in")
    frames_out = os.path.join(base_dir, f"{base_name}_frames_out")

    os.makedirs(frames_in, exist_ok=True)
    os.makedirs(frames_out, exist_ok=True)

    # --- Extract FPS and resolution from input video ---
    vidcap_in = cv2.VideoCapture(input_path)
    fps = vidcap_in.get(cv2.CAP_PROP_FPS) or 25  # fallback if FPS missing
    orig_w = int(vidcap_in.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(vidcap_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vidcap_in.release()

    print(f"[INFO] Input video properties → FPS={fps}, Size={orig_w}x{orig_h}")

    # --- Only extract frames if no outputs exist ---
    if not any(f.lower().endswith(".png") for f in os.listdir(frames_out)):
        print(f"[INFO] Extracting frames from {input_path} ...")
        if os.path.exists(frames_in):
            shutil.rmtree(frames_in)
        os.makedirs(frames_in, exist_ok=True)

        vidcap = cv2.VideoCapture(input_path)
        count = 0
        while True:
            success, frame = vidcap.read()
            if not success:
                break
            frame_path = os.path.join(frames_in, f"frame_{count:05d}.png")
            cv2.imwrite(frame_path, frame)
            count += 1
        vidcap.release()
        print(f"[INFO] Extracted {count} frames → {frames_in}")
    else:
        print(f"[CACHE] Detected existing frames in {frames_out}, skipping extraction.")

    try:
        # --- Run ComfyFlux pipeline on input frames ---
        #from Utils.flux import process_folder
        
        # process_folder(
        #     frames_in,
        #     frames_out,
        #     seed=seed,
        #     steps=steps,
        #     cfg=cfg,
        #     flux_guidance=flux_guidance,
        #     prompt_text=prompt_text
        # )

        prompt_text = (
    "restore and colorize with vivid colors, "
    "skin tones should be natural, realistic, "
    "colorful dresses (vivid colors) "        
    "and no warm or cool tint in the entire image")
        prompt_text = (
    "restore and colorize with vivid colors, "
    "skin tones should be natural, realistic, "      
    "and no warm or cool tint in the entire image")
        from Utils.qwen import process_folder
        process_folder(
        frames_in,
        frames_out,
        prompt_text,
        seed=123456789,   # 👈 your seed
        steps=5,         # 👈 number of denoising steps
        cfg=1.5)           # 👈 CFG guidance

        # --- Collect frames and build video ---
        out_files = sorted([os.path.join(frames_out, f) for f in os.listdir(frames_out)
                            if f.lower().endswith(".png")])
        if not out_files:
            raise RuntimeError("No frames generated by ComfyFlux!")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(output_path, fourcc, fps, (orig_w, orig_h))

        for f in out_files:
            img = cv2.imread(f)
            if img is None:
                continue
            img_resized = cv2.resize(img, (orig_w, orig_h))  # ensure exact size
            video.write(img_resized)
        video.release()

        print(f"[INFO] Saved comfyflux video → {output_path}")

    except (Exception, KeyboardInterrupt) as e:
        print(f"[ERROR] comfyflux colorization interrupted: {e}")
        if os.path.exists(output_path):
            print(f"[CLEANUP] Removing partial cached video: {output_path}")
            os.remove(output_path)
        raise
    finally:
        gc.collect()

    return output_path




def comfyflux_colorize_video_cached_old(input_path,
                                    first_path,
                                    prompt_text,
                                    seed=2^24,
                                    steps=10,
                                    cfg=1.0,
                                    flux_guidance=2.5):
    """
    ComfyFlux-based video colorization with caching and resume support.
    - Extracts frames only if no output frames exist.
    - Skips already processed frames on resume.
    - Combines frames into cached .mp4 using input video's FPS and size.
    """

    # --- Cached output path (.mp4) ---
    output_path = get_cached_file(input_path, "comfyflux_colorize_old", video_source_path=first_path) 
    if os.path.exists(output_path):
        print(f"[CACHE] Found comfyflux video: {output_path}")
        return output_path

    # --- Prepare input/output frames folders ---
    base_dir = pathlib.Path(output_path).parent
    base_name = pathlib.Path(output_path).stem
    frames_in = os.path.join(base_dir, f"{base_name}_frames_in")
    frames_out = os.path.join(base_dir, f"{base_name}_frames_out")

    os.makedirs(frames_in, exist_ok=True)
    os.makedirs(frames_out, exist_ok=True)

    # --- Extract FPS and resolution from input video ---
    vidcap_in = cv2.VideoCapture(input_path)
    fps = vidcap_in.get(cv2.CAP_PROP_FPS) or 25  # fallback if FPS missing
    orig_w = int(vidcap_in.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(vidcap_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vidcap_in.release()

    print(f"[INFO] Input video properties → FPS={fps}, Size={orig_w}x{orig_h}")

    # --- Only extract frames if no outputs exist ---
    if not any(f.lower().endswith(".png") for f in os.listdir(frames_out)):
        print(f"[INFO] Extracting frames from {input_path} ...")
        if os.path.exists(frames_in):
            shutil.rmtree(frames_in)
        os.makedirs(frames_in, exist_ok=True)

        vidcap = cv2.VideoCapture(input_path)
        count = 0
        while True:
            success, frame = vidcap.read()
            if not success:
                break
            frame_path = os.path.join(frames_in, f"frame_{count:05d}.png")
            cv2.imwrite(frame_path, frame)
            count += 1
        vidcap.release()
        print(f"[INFO] Extracted {count} frames → {frames_in}")
    else:
        print(f"[CACHE] Detected existing frames in {frames_out}, skipping extraction.")

    try:
        # --- Run ComfyFlux pipeline on input frames ---
        from Utils.flux import process_folder
        
        process_folder(
            frames_in,
            frames_out,
            seed=seed,
            steps=steps,
            cfg=cfg,
            flux_guidance=flux_guidance,
            prompt_text=prompt_text
        )

    #     prompt_text = (
    # "restore and colorize with vivid colors, "
    # "skin tones should be natural, realistic, "
    # "and no warm or cool tint in the entire image")
    #     from Utils.qwen import process_folder
    #     process_folder(
    #     frames_in,
    #     frames_out,
    #     prompt_text,
    #     seed=123456789,   # 👈 your seed
    #     steps=5,         # 👈 number of denoising steps
    #     cfg=1.5)           # 👈 CFG guidance

        # --- Collect frames and build video ---
        out_files = sorted([os.path.join(frames_out, f) for f in os.listdir(frames_out)
                            if f.lower().endswith(".png")])
        if not out_files:
            raise RuntimeError("No frames generated by ComfyFlux!")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(output_path, fourcc, fps, (orig_w, orig_h))

        for f in out_files:
            img = cv2.imread(f)
            if img is None:
                continue
            img_resized = cv2.resize(img, (orig_w, orig_h))  # ensure exact size
            video.write(img_resized)
        video.release()

        print(f"[INFO] Saved comfyflux video → {output_path}")

    except (Exception, KeyboardInterrupt) as e:
        print(f"[ERROR] comfyflux colorization interrupted: {e}")
        if os.path.exists(output_path):
            print(f"[CLEANUP] Removing partial cached video: {output_path}")
            os.remove(output_path)
        raise
    finally:
        gc.collect()

    return output_path


import os, cv2, gc, pathlib, shutil
from Utils.flux import process_folder_concat_split


def comfyflux_colorize_video_concat_cached(input_path,
                                           first_path,
                                           prompt_text,
                                           seed=2**24,
                                           steps=10,
                                           cfg=1.0,
                                           flux_guidance=2.5,
                                           images_per_row=2,
                                           total_images_per_combined=6):
    """
    ComfyFlux-based video colorization (concatenated 6-frame batch version).
    - Groups frames into combined grids (e.g. 6 per image).
    - Colorizes with ComfyFlux.
    - Splits and rebuilds colorized .mp4 output.
    """

    # --- Cached output path (.mp4) ---
    output_path = get_cached_file(input_path, "comfyflux_colorize_concat", video_source_path=first_path)
    if os.path.exists(output_path):
        print(f"[CACHE] Found comfyflux concatenated video: {output_path}")
        return output_path

    # --- Prepare frame directories ---
    base_dir = pathlib.Path(output_path).parent
    base_name = pathlib.Path(output_path).stem
    frames_in = os.path.join(base_dir, f"{base_name}_frames_in")
    frames_out = os.path.join(base_dir, f"{base_name}_frames_out")
    os.makedirs(frames_in, exist_ok=True)
    os.makedirs(frames_out, exist_ok=True)

    # --- Extract FPS and resolution ---
    vidcap_in = cv2.VideoCapture(input_path)
    fps = vidcap_in.get(cv2.CAP_PROP_FPS) or 25
    orig_w = int(vidcap_in.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(vidcap_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vidcap_in.release()
    print(f"[INFO] Input video → FPS={fps}, Size={orig_w}x{orig_h}")

    # --- Extract frames only if not already present ---
    if not any(f.lower().endswith(".png") for f in os.listdir(frames_out)):
        print(f"[INFO] Extracting frames from {input_path} ...")
        if os.path.exists(frames_in):
            shutil.rmtree(frames_in)
        os.makedirs(frames_in, exist_ok=True)

        vidcap = cv2.VideoCapture(input_path)
        count = 0
        while True:
            success, frame = vidcap.read()
            if not success:
                break
            frame_path = os.path.join(frames_in, f"frame_{count:05d}.png")
            cv2.imwrite(frame_path, frame)
            count += 1
        vidcap.release()
        print(f"[INFO] Extracted {count} frames → {frames_in}")
    else:
        print(f"[CACHE] Frames already extracted, skipping.")

    try:
        # --- Run ComfyFlux batch processing ---
        process_folder_concat_split(
            input_folder=frames_in,
            output_folder=frames_out,
            seed=seed,
            steps=steps,
            cfg=cfg,
            flux_guidance=flux_guidance,
            prompt_text=prompt_text,
            images_per_row=images_per_row,
            total_images_per_combined=total_images_per_combined
        )

        # --- Combine frames back into video ---
        out_files = sorted([
            os.path.join(frames_out, f)
            for f in os.listdir(frames_out)
            if f.lower().endswith(".png")
        ])
        if not out_files:
            raise RuntimeError("No colorized frames found!")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(output_path, fourcc, fps, (orig_w, orig_h))

        for f in out_files:
            img = cv2.imread(f)
            if img is None:
                continue
            img_resized = cv2.resize(img, (orig_w, orig_h))
            video.write(img_resized)
        video.release()

        print(f"[INFO] Saved comfyflux concatenated video → {output_path}")

    except (Exception, KeyboardInterrupt) as e:
        print(f"[ERROR] comfyflux batch colorization interrupted: {e}")
        if os.path.exists(output_path):
            print(f"[CLEANUP] Removing partial cached video: {output_path}")
            os.remove(output_path)
        raise
    finally:
        gc.collect()

    return output_path




import os, cv2, gc, pathlib, shutil, math, re
import numpy as np
from tqdm import tqdm
from PIL import Image
from Utils.flux import process_scene_batches, process_scene_batches_without_overlap



def comfyflux_colorize_video_concat_scene_batch_cached(
    input_video_path,
    first_path,
    prompt_text,
    seed=2**24,
    steps=10,
    cfg=1.0,
    flux_guidance=2.5,
    images_per_row=2,
    total_images_per_combined=6
):
    """
    Scene-wise ComfyFlux batching.
    Uses existing frames (e.g. restore_bw_film-Scene-001_m_(PrevScene1).jpg)
    in the same folder as the input video.
    Detects scenes by '(PrevScene#)' grouping, batches frames per scene,
    colorizes, and concatenates all outputs chronologically.
    """
    print_flag =False
    # --- 1. Locate frame folder ---
    folder = os.path.dirname(input_video_path)
    print(f"[INFO] Using frame folder: {folder}")

    # --- 2. Define cached output video path ---
    output_path = get_cached_file(input_video_path, "comfyflux_scene_batch", video_source_path=first_path)
    if os.path.exists(output_path):
        print(f"[CACHE] Found comfyflux scene-batch video: {output_path}")
        return output_path

    # --- 3. Prepare output directory ---
    output_root = os.path.join(folder, "comfyflux_scene_outputs")
    os.makedirs(output_root, exist_ok=True)

    # --- 4. Collect all frame files ---
    all_files = sorted([f for f in os.listdir(folder)
                        if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    if not all_files:
        raise ValueError(f"❌ No frame images found in {folder}")

    sample = cv2.imread(os.path.join(folder, all_files[0]))
    if sample is None:
        raise ValueError("❌ Cannot read sample frame for dimension check.")
    orig_h, orig_w = sample.shape[:2]
    fps = 25  # fallback if FPS missing

    # --- 5. Group by (PrevScene#) ---
    prev_scene_pattern = re.compile(r"PrevScene(\d+)")
    scenes = {}
    for f in all_files:
        match = prev_scene_pattern.search(f)
        scene_key = f"PrevScene{match.group(1)}" if match else "PrevScene_Unknown"
        scenes.setdefault(scene_key, []).append(f)

    # --- 6. Sort scenes numerically ---
    def extract_scene_num(name: str) -> int:
        match = re.search(r"PrevScene(\d+)", name)
        return int(match.group(1)) if match else 9999

    ordered_scenes = sorted(scenes.items(), key=lambda kv: extract_scene_num(kv[0]))
    #print(f"[INFO] Found {len(ordered_scenes)} scene groups → {[s[0] for s in ordered_scenes]}")

    # --- 7. Process each scene sequentially ---
   # for scene_idx, (scene_name, frames) in enumerate(ordered_scenes, start=1):
    from tqdm import tqdm
    # tqdm for scene-level progress
    for scene_idx, (scene_name, frames) in tqdm(enumerate(ordered_scenes, start=1), total=len(ordered_scenes), desc="Scenes", unit="scene"):
        if(print_flag):
           print(f"\n🎬 Processing {scene_name} ({len(frames)} frames)")
        scene_out = os.path.join(output_root, scene_name)
        os.makedirs(scene_out, exist_ok=True)

        process_scene_batches_without_overlap(
            input_folder=folder,
            frames=frames,
            output_folder=scene_out,
            seed=seed,
            steps=steps,
            cfg=cfg,
            flux_guidance=flux_guidance,
            prompt_text=prompt_text,
            images_per_row=images_per_row,
            total_images_per_combined=total_images_per_combined
        )

    # --- 8. Concatenate all scene outputs in numeric order ---
    scene_dirs = [
        d for d in os.listdir(output_root)
        if os.path.isdir(os.path.join(output_root, d))
    ]

    def extract_scene_num2(name: str) -> int:
        m = re.search(r"PrevScene(\d+)", name)
        return int(m.group(1)) if m else 9999

    scene_dirs = sorted(scene_dirs, key=extract_scene_num2)
    #print(f"[INFO] Concatenating scenes in order: {scene_dirs}")

    all_out_files = []
    for scene_dir in scene_dirs:
        scene_path = os.path.join(output_root, scene_dir)
        frame_files = sorted([
            os.path.join(scene_path, f)
            for f in os.listdir(scene_path)
            if f.lower().endswith(".png")
        ])
        all_out_files.extend(frame_files)

    if not all_out_files:
        raise RuntimeError("❌ No frames found in any scene output folder!")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(output_path, fourcc, fps, (orig_w, orig_h))

    for f in all_out_files:
        img = cv2.imread(f)
        if img is not None:
            video.write(cv2.resize(img, (orig_w, orig_h)))

    video.release()
    print(f"[✅] Saved comfyflux colorized video → {output_path}")

    gc.collect()
    return output_path






def deoldify_cached(input_path, first_path):
    """Upscale video using ONNX, using filecache to avoid reprocessing."""


    input_path = Path(input_path)
    output_path = input_path.parent / "color_deoldify.mp4"
    if output_path.exists():
        print(f"[CACHE] Colorized output already exists: {output_path}")
        return str(output_path)
    from Utils.deoldify_utils import deoldify_parallel
    deoldify_parallel(input_path, output_path, num_workers=6 if torch.cuda.is_available() else 1)
    # from main_utils import repair_video_file
    # repair_video_file(output_path)
    return output_path

    
def run_colorization_cached(input_path: str, generator_path: str, first_path: str) -> str:
    """
    Run UNet-based colorization on the given grayscale video (input_path).
    Saves the colorized output as 'color.mp4' in the same directory.
    
    Parameters:
        input_path: Path to grayscale video (e.g., *_bw.mp4).
        generator_path: Path to UNet weights/model file.
    
    Returns:
        Full path to the colorized video (color.mp4).
    """

    output_video_path = get_cached_file(input_path, "unet_colored_refs", video_source_path=first_path)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"[ERROR] Grayscale video not found at: {input_path}")

    if os.path.exists(output_video_path):
        print(f"[CACHE] Colorized output already exists: {output_video_path}")
        return str(output_video_path)

    print("[INFO] Running UNET colorisation...")
    from Utils.unet_utils import load_generator_for_inference, recolor_video_with_external_luminance
    try:
        recolor_video_with_external_luminance(
            generator_path,
            str(input_path),
            str(input_path),
            str(output_video_path),
            frame_size=(256, 256)
        )
    except (Exception, KeyboardInterrupt) as e:
        print(f"[ERROR] UNET Colorisation interrupted: {e}")
        if os.path.exists(output_video_path):
            os.remove(output_video_path)
        cache_folder = os.path.join(os.path.dirname(input_path), "cache")
        if os.path.exists(cache_folder) and os.path.isdir(cache_folder):
            print(f"[CLEANUP] Removing cache folder: {cache_folder}")
            shutil.rmtree(cache_folder)
        raise

    from Utils.main_utils import repair_video_file
    repair_video_file(str(output_video_path))
    print(f"[DONE] Colorization completed with external Luminance→ {output_video_path}")
    return str(output_video_path)


def run_unet_colorization_cached_subprocess(input_bw_video, unet_weights, first_path):
    """
    Runs UNet-based colorization via subprocess with caching logic.
    This wraps a subprocess call to `run_colorization_cached()` in Utils.main_utils.

    Parameters:
        input_bw_video: path to grayscale video (e.g., *_bw.mp4)
        unet_weights: path to UNet weights
        first_path: root/original video path (used for cache folder naming)

    Returns:
        Path to the final colorized video
    """

    # Free up GPU memory
    import torch, gc
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()

    output_video_path = get_cached_file(input_bw_video, "unet_colored_refs", video_source_path=first_path)
    if os.path.exists(output_video_path):
        print(f"[CACHE] Colorized output already exists: {output_video_path}")
        return str(output_video_path)
    print("[INFO] Launching UNET colorization subprocess...")

    try:
        subprocess.run([
            sys.executable, "-c",
            (
                "from Utils.main_utils import run_colorization_cached; "
                f"run_colorization_cached('{input_bw_video}', '{unet_weights}', '{first_path}')"
            )
        ], check=True)
        return output_video_path

    except (subprocess.CalledProcessError, Exception, KeyboardInterrupt) as e:
        print(f"[ERROR] UNET Colorisation subprocess failed: {e}")

        # Remove output video if partially created
        if os.path.exists(output_video_path):
            print(f"[CLEANUP] Removing partial output video: {output_video_path}")
            os.remove(output_video_path)

        # Remove related cache folder
        cache_folder = os.path.join(os.path.dirname(input_bw_video), "cache")
        if os.path.exists(cache_folder) and os.path.isdir(cache_folder):
            print(f"[CLEANUP] Removing cache folder: {cache_folder}")
            shutil.rmtree(cache_folder)
        raise
    finally:
        import torch, gc
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
    return output_video_path



def enhance_unet_cached(input_path, first_path):
    """Upscaling Faces, using filecache to avoid reprocessing."""
    output_path = get_cached_file(input_path, "unet_enhanced", video_source_path=first_path)
    if os.path.exists(output_path):
        print(f"[CACHE] UNET enhanced video found: {output_path}")
        return output_path
    print("[INFO] Running Diffusion to enhance unet output...")
    from Utils.diffusion_unet_enhance_utils import diffusion_unet_enhance
    try: 
        diffusion_unet_enhance(input_path, output_path)
    except (Exception, KeyboardInterrupt) as e:
        print(f"[ERROR] Diffusion enhancing interrupted: {e}")
        if os.path.exists(output_path):
            print(f"[CLEANUP] Removing partial output video: {output_path}")
            os.remove(output_path)
        raise
    finally:
        import torch, gc
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
    return output_path





# import os
# import cv2
# import torch
# import pickle
# import numpy as np
# from tqdm import tqdm
# from pathlib import Path
# from ultralytics import YOLO

# # ---- Config ----
# YOLO_MODEL_PATH = "models/yolo11x-seg.pt"
# CONF_THRESHOLD = 0.6
# device_str = "cuda" if torch.cuda.is_available() else "cpu"

# # Load YOLO segmentation model once
# yolo_model = YOLO(YOLO_MODEL_PATH)


# def replace_masked_regions_between_videos(source_video, target_video, output_suffix="_maskedmerge.mp4"):
#     """
#     Run YOLO segmentation on source_video and replace only masked areas onto target_video.
#     The output is saved parallel to target_video with a custom suffix.

#     Args:
#         source_video (str): Path to source video (YOLO will run here)
#         target_video (str): Path to target video (masked regions replaced here)
#         output_suffix (str): Suffix for the generated output video filename

#     Returns:
#         str: Path to saved output video
#     """
#     # Define output folder and cache
#     target_path = Path(target_video)
#     out_dir = target_path.parent / "yolo_cache"
#     out_dir.mkdir(parents=True, exist_ok=True)

#     # --- Step 1: Run YOLO segmentation ---
#     yolo_cache_path = out_dir / f"{target_path.stem}_yolo_masks.pkl"
#     if yolo_cache_path.exists():
#         with open(yolo_cache_path, "rb") as f:
#             masks_per_frame = pickle.load(f)
#     else:
#         masks_per_frame = []
#         cap = cv2.VideoCapture(source_video)
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         with tqdm(total=total_frames, desc="Running YOLO segmentation", unit="frame") as pbar:
#             while True:
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
#                 results = yolo_model.predict(frame, conf=CONF_THRESHOLD, verbose=False, device=device_str)
#                 if len(results[0].masks):
#                     masks = results[0].masks.data.cpu().numpy()
#                 else:
#                     masks = []
#                 masks_per_frame.append(masks)
#                 pbar.update(1)
#         cap.release()
#         with open(yolo_cache_path, "wb") as f:
#             pickle.dump(masks_per_frame, f)

#     # --- Step 2: Apply masks ---
#     cap_src = cv2.VideoCapture(source_video)
#     cap_tgt = cv2.VideoCapture(target_video)
#     fps = cap_src.get(cv2.CAP_PROP_FPS)
#     width = int(cap_src.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap_src.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     total_frames = int(cap_src.get(cv2.CAP_PROP_FRAME_COUNT))

#     # Create output video path (parallel to target)
#     output_path = str(target_path.with_name(target_path.stem + output_suffix))
#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#     with tqdm(total=total_frames, desc="Merging masked regions", unit="frame") as pbar:
#         for i in range(total_frames):
#             ret_src, frame_src = cap_src.read()
#             ret_tgt, frame_tgt = cap_tgt.read()
#             if not (ret_src and ret_tgt):
#                 break

#             frame_result = frame_tgt.copy()

#             if i < len(masks_per_frame) and len(masks_per_frame[i]) > 0:
#                 for mask in masks_per_frame[i]:
#                     mask_resized = cv2.resize(mask, (width, height))
#                     mask_binary = (mask_resized > 0.5).astype(np.uint8)
#                     mask_3ch = np.repeat(mask_binary[:, :, None], 3, axis=2)
#                     frame_result[mask_3ch == 1] = frame_src[mask_3ch == 1]

#             out.write(frame_result)
#             pbar.update(1)

#     cap_src.release()
#     cap_tgt.release()
#     out.release()

#     print(f"✅ Output saved parallel to target: {output_path}")
#     return output_path





import os
import cv2
import torch
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO

# ---- Config ----
YOLO_MODEL_PATH = "models/yolo11x-seg.pt"
CONF_THRESHOLD = 0.40
device_str = "cuda" if torch.cuda.is_available() else "cpu"

# Load YOLO segmentation model once
yolo_model = YOLO(YOLO_MODEL_PATH)


def replace_masked_regions_between_videos(source_video, target_video, output_suffix="_maskedmerge.mp4", class_filter="person"):
    """
    Run YOLO segmentation on source_video and replace only masked areas of the given class
    (default: 'person') onto target_video.
    The output is saved parallel to target_video with a custom suffix.
    """
    target_path = Path(target_video)
    out_dir = target_path.parent / "yolo_cache"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Run YOLO segmentation ---
    yolo_cache_path = out_dir / f"{target_path.stem}_yolo_masks.pkl"
    if yolo_cache_path.exists():
        with open(yolo_cache_path, "rb") as f:
            masks_per_frame = pickle.load(f)
    else:
        masks_per_frame = []
        cap = cv2.VideoCapture(source_video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        with tqdm(total=total_frames, desc="Running YOLO segmentation", unit="frame") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                results = yolo_model.predict(frame, conf=CONF_THRESHOLD, verbose=False, device=device_str)

                masks = []
                if len(results) > 0 and len(results[0].boxes) > 0:
                    names = results[0].names
                    for j, cls_id in enumerate(results[0].boxes.cls.cpu().numpy()):
                        label = names[int(cls_id)]
                        if class_filter is None or label == class_filter:
                            if results[0].masks is not None:
                                masks.append(results[0].masks.data[j].cpu().numpy())
                masks_per_frame.append(masks)
                pbar.update(1)
        cap.release()
        with open(yolo_cache_path, "wb") as f:
            pickle.dump(masks_per_frame, f)

    # --- Step 2: Apply masks ---
    cap_src = cv2.VideoCapture(source_video)
    cap_tgt = cv2.VideoCapture(target_video)
    fps = cap_src.get(cv2.CAP_PROP_FPS)
    width = int(cap_src.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_src.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap_src.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output path parallel to target
    output_path = str(target_path.with_name(target_path.stem + output_suffix))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    with tqdm(total=total_frames, desc="Merging masked regions", unit="frame") as pbar:
        for i in range(total_frames):
            ret_src, frame_src = cap_src.read()
            ret_tgt, frame_tgt = cap_tgt.read()
            if not (ret_src and ret_tgt):
                break

            frame_result = frame_tgt.copy()

            # If frame has valid masks (e.g., persons), apply them
            if i < len(masks_per_frame) and len(masks_per_frame[i]) > 0:
                for mask in masks_per_frame[i]:
                    mask_resized = cv2.resize(mask, (width, height))
                    mask_binary = (mask_resized > 0.5).astype(np.uint8)
                    mask_3ch = np.repeat(mask_binary[:, :, None], 3, axis=2)
                    frame_result[mask_3ch == 1] = frame_src[mask_3ch == 1]
            # else: no persons → keep target frame as is

            out.write(frame_result)
            pbar.update(1)

    cap_src.release()
    cap_tgt.release()
    out.release()

    print(f"✅ Output saved parallel to target: {output_path}")
    return output_path






def replace_masked_regions_between_videos(
    source_video,
    target_video,
    output_suffix="_maskedmerge.mp4",
    class_filter="person"
):
    """
    Run YOLO segmentation on source_video and replace only masked areas of the given class
    (default: 'person') onto target_video.

    Outputs:
      1. *_yolo_segmentation.mp4 → visualization directly from YOLO `results[0].plot()`
      2. *_maskedmerge.mp4 → merged masked regions from source→target

    Returns:
      str → path to masked merge video (safe to use directly)
    """

    import os, cv2, pickle
    import numpy as np
    from pathlib import Path
    from tqdm import tqdm

    target_path = Path(target_video)
    out_dir = target_path.parent / "yolo_cache"
    out_dir.mkdir(parents=True, exist_ok=True)

    masked_output_path = str(target_path.with_name(target_path.stem + output_suffix))
    seg_output_path = str(target_path.with_name(target_path.stem + "_yolo_segmentation.mp4"))

    print(f"[INFO] Output will be saved:")
    print(f"   🎬 Masked merge  → {masked_output_path}")
    print(f"   🎨 Segmentation → {seg_output_path}")

    # --- Step 1: YOLO segmentation ---
    yolo_cache_path = out_dir / f"{target_path.stem}_yolo_masks.pkl"

    if yolo_cache_path.exists():
        print(f"[CACHE] Loading cached YOLO masks → {yolo_cache_path}")
        with open(yolo_cache_path, "rb") as f:
            masks_per_frame = pickle.load(f)
    else:
        print(f"[INFO] Running YOLO segmentation on {source_video}")
        masks_per_frame = []
        cap = cv2.VideoCapture(source_video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        seg_writer = cv2.VideoWriter(seg_output_path, fourcc, fps, (width, height))

        with tqdm(total=total_frames, desc="Running YOLO segmentation", unit="frame") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Run YOLO prediction
                results = yolo_model.predict(frame, conf=CONF_THRESHOLD, verbose=False, device=device_str)
                vis_frame = frame.copy()
                masks = []

                if len(results) > 0:
                    # --- YOLO’s built-in plot() for automatic overlays ---
                    vis_frame = results[0].plot()

                    # Collect masks for class_filter
                    if results[0].masks is not None:
                        names = results[0].names
                        for j, cls_id in enumerate(results[0].boxes.cls.cpu().numpy()):
                            label = names[int(cls_id)]
                            if class_filter is None or label == class_filter:
                                masks.append(results[0].masks.data[j].cpu().numpy())

                seg_writer.write(vis_frame)
                masks_per_frame.append(masks)
                pbar.update(1)

        cap.release()
        seg_writer.release()

        # Cache YOLO masks for re-use
        with open(yolo_cache_path, "wb") as f:
            pickle.dump(masks_per_frame, f)

        print(f"[CACHE] YOLO masks saved → {yolo_cache_path}")
        print(f"🎨 Segmentation video saved → {seg_output_path}")

    # --- Step 2: Apply masks for merging ---
    cap_src = cv2.VideoCapture(source_video)
    cap_tgt = cv2.VideoCapture(target_video)

    fps = cap_src.get(cv2.CAP_PROP_FPS)
    width = int(cap_src.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_src.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap_src.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(masked_output_path, fourcc, fps, (width, height))

    print(f"[INFO] Starting masked merge ({class_filter}) for {total_frames} frames...")
    with tqdm(total=total_frames, desc="Merging masked regions", unit="frame") as pbar:
        for i in range(total_frames):
            ret_src, frame_src = cap_src.read()
            ret_tgt, frame_tgt = cap_tgt.read()
            if not (ret_src and ret_tgt):
                break

            frame_result = frame_tgt.copy()

            # Apply YOLO masks (if present)
            if i < len(masks_per_frame) and len(masks_per_frame[i]) > 0:
                for mask in masks_per_frame[i]:
                    mask_resized = cv2.resize(mask, (width, height))
                    mask_binary = (mask_resized > 0.5).astype(np.uint8)
                    mask_3ch = np.repeat(mask_binary[:, :, None], 3, axis=2)
                    frame_result[mask_3ch == 1] = frame_src[mask_3ch == 1]

            out.write(frame_result)
            pbar.update(1)

    cap_src.release()
    cap_tgt.release()
    out.release()

    print(f"✅ Masked merge completed successfully.")
    print(f"🎬 Output videos saved:")
    print(f"   1️⃣ Segmentation video → {seg_output_path}")
    print(f"   2️⃣ Masked merge video → {masked_output_path}")

    # --- return only the final video path ---
    return masked_output_path




import os
import cv2
import torch
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.ops import scale_masks


def replace_masked_regions_between_videos(
    source_video,
    target_video,
    output_suffix="_maskedmerge.mp4",
    class_filter="person"
):
    """
    Run YOLO segmentation on source_video and replace only masked areas of the given class
    (default: 'person') onto target_video.

    Outputs:
      1. *_yolo_segmentation.mp4 → YOLO segmentation visualization
      2. *_maskedmerge.mp4 → masked merge result
    """

    # =========================================================
    # STEP 1: Setup Paths and Model
    # =========================================================
    target_path = Path(target_video)
    out_dir = target_path.parent / "yolo_cache"
    out_dir.mkdir(parents=True, exist_ok=True)

    masked_output_path = str(target_path.with_name(target_path.stem + output_suffix))
    seg_output_path = str(target_path.with_name(target_path.stem + "_yolo_segmentation.mp4"))
    yolo_cache_path = out_dir / f"{target_path.stem}_yolo_masks.pkl"

    print(f"[INFO] Outputs:")
    print(f"   🎨 Segmentation → {seg_output_path}")
    print(f"   🎬 Masked merge → {masked_output_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO("yolo11x-seg.pt")
    model.to(device)
    print(f"[INFO] Model loaded on {device.upper()}")

    # =========================================================
    # STEP 2: Run YOLO Segmentation or Load Cached Masks
    # =========================================================
    if yolo_cache_path.exists():
        print(f"[CACHE] Loading cached YOLO masks → {yolo_cache_path}")
        with open(yolo_cache_path, "rb") as f:
            masks_per_frame = pickle.load(f)
    else:
        print(f"[INFO] Running YOLO segmentation on {source_video}")
        cap = cv2.VideoCapture(source_video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        seg_writer = cv2.VideoWriter(seg_output_path, fourcc, fps, (width, height))

        masks_per_frame = []

        with tqdm(total=total_frames, desc="YOLO Segmentation", unit="frame") as pbar:
            for _ in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame, verbose=False)
                result = results[0]
                vis_frame = result.plot()

                # Collect masks for chosen class
                masks = []
                if result.masks is not None:
                    names = result.names
                    for i, box in enumerate(result.boxes):
                        label = names[int(box.cls)]
                        if class_filter is None or label == class_filter:
                            mask_tensor = result.masks.data[i].float()
                            mask_resized = scale_masks(mask_tensor[None, None], (height, width))[0, 0] > 0.5
                            masks.append(mask_resized.cpu().numpy().astype(np.uint8))

                seg_writer.write(vis_frame)
                masks_per_frame.append(masks)
                pbar.update(1)

        cap.release()
        seg_writer.release()

        # Cache YOLO masks
        with open(yolo_cache_path, "wb") as f:
            pickle.dump(masks_per_frame, f)

        print(f"[CACHE] YOLO masks saved → {yolo_cache_path}")
        print(f"🎨 Segmentation video saved → {seg_output_path}")

    # =========================================================
    # STEP 3: Merge Masked Regions
    # =========================================================
    cap_src = cv2.VideoCapture(source_video)
    cap_tgt = cv2.VideoCapture(target_video)
    fps = int(cap_src.get(cv2.CAP_PROP_FPS))
    width = int(cap_src.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_src.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(min(
        cap_src.get(cv2.CAP_PROP_FRAME_COUNT),
        cap_tgt.get(cv2.CAP_PROP_FRAME_COUNT)
    ))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(masked_output_path, fourcc, fps, (width, height))

    print(f"[INFO] Merging masked regions for {total_frames} frames...")
    with tqdm(total=total_frames, desc="Merging", unit="frame") as pbar:
        for i in range(total_frames):
            ret_src, frame_src = cap_src.read()
            ret_tgt, frame_tgt = cap_tgt.read()
            if not (ret_src and ret_tgt):
                break

            result_frame = frame_tgt.copy()
            if i < len(masks_per_frame) and masks_per_frame[i]:
                for mask in masks_per_frame[i]:
                    mask_resized = cv2.resize(mask, (width, height))
                    mask_3ch = np.repeat(mask_resized[:, :, None], 3, axis=2)
                    result_frame[mask_3ch == 1] = frame_src[mask_3ch == 1]

            out_writer.write(result_frame)
            pbar.update(1)

    cap_src.release()
    cap_tgt.release()
    out_writer.release()

    print(f"\n✅ Masked merge completed successfully.")
    print(f"🎬 Outputs:")
    print(f"   1️⃣ Segmentation video → {seg_output_path}")
    print(f"   2️⃣ Masked merge video → {masked_output_path}")

    return masked_output_path






def colorize_scenes_cached(scene_refs_video,  before_path, colorized_scene_path, first_path):
    """Propagating colors, using filecache to avoid reprocessing."""
    output_path = get_cached_file(colorized_scene_path, "colored_full_video", video_source_path=first_path)
    if os.path.exists(output_path):
        print(f"[CACHE] Propagated colored full video found: {output_path}")
        return output_path
    print("[INFO] Running Color propagation...")
    from Utils.color_propagation import colorize_scenes
    try:
        colorize_scenes(scene_refs_video,  before_path, colorized_scene_path, output_path)
    except (Exception, KeyboardInterrupt) as e:
        print(f"[ERROR] color propagation interrupted: {e}")
        if os.path.exists(output_path):
            print(f"[CLEANUP] Removing partial output video: {output_path}")
            os.remove(output_path)
        # output_folder =  os.path.dirname(output_path) 
        # if os.path.exists(output_folder) and os.path.isdir(output_folder):
        #     print(f"[CLEANUP] Removing output folder: {output_folder}")
        #     shutil.rmtree(output_folder)
        raise
    finally:
        import torch, gc
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
    return output_path



def colorize_scenes_prev_cached(scene_refs_video,  before_path, colorized_scene_path, first_path):
    """Propagating colors, using filecache to avoid reprocessing."""
    output_path = get_cached_file(colorized_scene_path, "colored_full_video", video_source_path=first_path)
    if os.path.exists(output_path):
        print(f"[CACHE] Propagated colored full video found: {output_path}")
        return output_path
    print("[INFO] Running Color propagation...")
    from Utils.color_propagation import colorize_scenes_prev
    try:
        colorize_scenes_prev(scene_refs_video,  before_path, colorized_scene_path, output_path)
    except (Exception, KeyboardInterrupt) as e:
        print(f"[ERROR] color propagation interrupted: {e}")
        if os.path.exists(output_path):
            print(f"[CLEANUP] Removing partial output video: {output_path}")
            os.remove(output_path)
        # output_folder =  os.path.dirname(output_path) 
        # if os.path.exists(output_folder) and os.path.isdir(output_folder):
        #     print(f"[CLEANUP] Removing output folder: {output_folder}")
        #     shutil.rmtree(output_folder)
        raise
    finally:
        import torch, gc
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
    return output_path



    
def postprocess_videos_cached(input_path, first_path):

    output_path = get_cached_file(input_path, "postprocess", video_source_path=first_path)
    if os.path.exists(output_path):
        print(f"[CACHE] postprocessed file found: {output_path}")
        return output_path

    print("[INFO] Running postprocess")
    from Utils.postprocess_videos_utils import enhance_saturation_and_brightness
    try:
        enhance_saturation_and_brightness(input_path, output_path)
    except (Exception, KeyboardInterrupt) as e:
        print(f"[ERROR] postprocess interrupted: {e}")
        if os.path.exists(output_path):
            print(f"[CLEANUP] Removing partial output video: {output_path}")
            os.remove(output_path)
        raise
    finally:
        import torch, gc
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
    print(f"[DONE] postprocess video created at: {output_path}")
    return output_path


def remix_audio_cached(input_path, first_path, output_string):

    output_path = get_cached_file(input_path, output_string, video_source_path=first_path)
    if os.path.exists(output_path):
        print(f"[CACHE] Final video found: {output_path}")
        return output_path

    print("[INFO] Running Audio Remix")
    from Utils.remix_audio_utils import remix_audio
    try:
        remix_audio(input_path, first_path, output_path)
    except (Exception, KeyboardInterrupt) as e:
        print(f"[ERROR] Audio Remix interrupted: {e}")
        if os.path.exists(output_path):
            print(f"[CLEANUP] Removing partial output video: {output_path}")
            os.remove(output_path)
        raise
    finally:
        import torch, gc
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
    print(f"[DONE] Audio Remix created at: {output_path}")
    return output_path

def concat_videos_cached(input_path, color_path):

    output_path = get_cached_file(color_path, "concated", video_source_path=input_path)
    if os.path.exists(output_path):
        print(f"[CACHE] concated file found: {output_path}")
        return output_path

    print("[INFO] Running concatenation")
    from Utils.concat_videos_utils import concat_video_frames
    concat_video_frames(input_path, color_path, output_path)
    from Utils.main_utils import repair_video_file
    repair_video_file(str(output_path))
    return output_path


import os

_original_stderr_fd = None  # Global to track original stderr

def suppress_cpp_stderr():
    """
    Suppress C++ backend stderr (e.g., TensorFlow Lite, MediaPipe).
    """
    global _original_stderr_fd
    if _original_stderr_fd is None:
        _original_stderr_fd = os.dup(2)  # Save original stderr
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, 2)  # Redirect stderr to null
        os.close(devnull_fd)

def restore_cpp_stderr():
    """
    Restore original stderr (for tqdm, logging, etc.)
    """
    global _original_stderr_fd
    if _original_stderr_fd is not None:
        os.dup2(_original_stderr_fd, 2)  # Restore original stderr
        os.close(_original_stderr_fd)
        _original_stderr_fd = None


