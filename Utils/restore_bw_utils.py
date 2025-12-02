import subprocess
from pathlib import Path
import contextlib
import sys
import re
import unicodedata
import os
import hashlib
import time
from functools import lru_cache
import shutil
from Utils.main_utils import get_frame_count
import json
from tqdm import tqdm



def get_video_fps(input_path):
    """Use ffprobe to extract actual FPS from input video."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "json",
        input_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(result.stdout)
    rate_str = info["streams"][0]["r_frame_rate"]  # e.g. "30000/1001"
    num, denom = map(int, rate_str.split("/"))
    return num / denom


def timestamp_to_seconds(h, m, s, ms):
    return h * 3600 + m * 60 + s + ms / 100.0


def restore_bw_film(input_path, output_path):
    """Restore B&W film using FFmpeg and show progress bar."""
    #Path("restored").mkdir(exist_ok=True)

    total_frames = get_frame_count(input_path)
    fps = get_video_fps(input_path)

    deflicker_size = min(max(total_frames // 4, 4), 16)
    tmix_frames = min(3, max(1, total_frames // 10))

    print(f"[INFO] Frame count: {total_frames}, fps: {fps:.2f}, deflicker size: {deflicker_size}, tmix frames: {tmix_frames}")

    vf_filters = (
        f"format=gray,"
        f"hqdn3d=2.5:1.5:3.0:3.0,"
        f"deflicker=mode=pm:size={deflicker_size},"
        f"eq=contrast=1.15:brightness=0.02,"
        f"scale=iw:ih,"
        f"unsharp=3:3:0.7:3:3:0.0"
    )

    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-vf", vf_filters,
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "22",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-ar", "44100",
        "-movflags", "+faststart",
        output_path,
    ]

    #print(f"[INFO] Running: {' '.join(cmd)}")

    process = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    time_pattern = re.compile(r'time=(\d+):(\d+):(\d+).(\d+)')

    progress_bar = tqdm(total=total_frames, desc="Restoring", unit="frame")
    last_frame = 0

    for line in process.stderr:
        match = time_pattern.search(line)
        if match:
            h, m, s, ms = map(int, match.groups())
            seconds = timestamp_to_seconds(h, m, s, ms)
            current_frame = int(fps * seconds)

            if current_frame > last_frame:
                progress_bar.update(current_frame - last_frame)
                last_frame = current_frame

    process.wait()
    progress_bar.update(total_frames - last_frame)  # ensure it completes
    progress_bar.close()

    if process.returncode != 0:
        print("[ERROR] FFmpeg failed during restoration.")
        return

    restored_frame_count = get_frame_count(output_path)
    #print(f"[INFO] Output frame count: {restored_frame_count}")

    if restored_frame_count < 5:
        print("[WARN] Restored output seems empty or black — copying original instead.")
        shutil.copy2(input_path, output_path)


def restore_bw_film_(input_path, output_path):
    """Restore B&W film using FFmpeg, then run remaster.py, final result at output_path."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    total_frames = get_frame_count(input_path)
    fps = get_video_fps(input_path)

    deflicker_size = min(max(total_frames // 4, 4), 16)
    tmix_frames = min(3, max(1, total_frames // 10))

    print(f"[INFO] Frame count: {total_frames}, fps: {fps:.2f}, "
          f"deflicker size: {deflicker_size}, tmix frames: {tmix_frames}")

    vf_filters = (
        f"format=gray,"
        f"hqdn3d=2.5:1.5:3.0:3.0,"
        f"deflicker=mode=pm:size={deflicker_size},"
        f"eq=contrast=1.15:brightness=0.02,"
        f"scale=iw:ih,"
        f"unsharp=3:3:0.7:3:3:0.0"
    )

    # --------------------------
    # Step 1: FFmpeg restoration
    # --------------------------
    restored_temp = Path(output_path).with_suffix(".intermediate.mp4")

    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-vf", vf_filters,
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "22",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-ar", "44100",
        "-movflags", "+faststart",
        str(restored_temp),
    ]

    print(f"[INFO] Running FFmpeg restoration: {' '.join(cmd)}")

    process = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    time_pattern = re.compile(r'time=(\d+):(\d+):(\d+).(\d+)')

    progress_bar = tqdm(total=total_frames, desc="FFmpeg Restore", unit="frame", unit_scale=True, smoothing=0.1)
    last_frame = 0

    for line in process.stderr:
        match = time_pattern.search(line)
        if match:
            h, m, s, ms = map(int, match.groups())
            seconds = timestamp_to_seconds(h, m, s, ms)
            current_frame = int(fps * seconds)

            if current_frame > last_frame:
                progress_bar.update(current_frame - last_frame)
                last_frame = current_frame

    process.wait()
    progress_bar.update(total_frames - last_frame)
    progress_bar.close()

    if process.returncode != 0:
        print("[ERROR] FFmpeg failed during restoration.")
        return

    restored_frame_count = get_frame_count(restored_temp)
    print(f"[INFO] Intermediate restored frame count: {restored_frame_count}")

    if restored_frame_count < 5:
        print("[WARN] Restored output seems empty — copying original instead.")
        shutil.copy2(input_path, output_path)
        return

    # --------------------------
    # Step 2: Run remaster.py
    # --------------------------
    print(f"[INFO] Running remaster.py, final output: {output_path}")

    remaster_cmd = [
        "python", "remaster.py",
        "--input", str(restored_temp),
        "--disable_colorization",
        "--gpu",
        "--output", str(output_path)
    ]

    print(f"[INFO] Running: {' '.join(remaster_cmd)}")

    # 👉 Let remaster.py show its own tqdm progress bar
    ret = subprocess.call(remaster_cmd)

    if ret != 0:
        print("[ERROR] Remastering step failed.")
    else:
        print(f"[INFO] Remastering completed. Final video saved at: {output_path}")

    # --------------------------
    # Cleanup
    # --------------------------
    try:
        restored_temp.unlink()
        print(f"[INFO] Removed intermediate file: {restored_temp}")
    except Exception:
        pass






import subprocess
import shutil
import re
from pathlib import Path
from tqdm import tqdm

def restore_bw_film_(input_path, output_path, gpu_id="0"):
    """
    Restore B&W film using FFmpeg pre-filtering, then run Microsoft's
    Bringing-Old-Photos-Back-to-Life pipeline with fixed parameters:
    --with_scratch and --HR.
    Final restored video is saved at output_path.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    total_frames = get_frame_count(input_path)
    fps = get_video_fps(input_path)

    deflicker_size = min(max(total_frames // 4, 4), 16)
    tmix_frames = min(3, max(1, total_frames // 10))

    print(f"[INFO] Frame count: {total_frames}, fps: {fps:.2f}, "
          f"deflicker size: {deflicker_size}, tmix frames: {tmix_frames}")

    vf_filters = (
        f"format=gray,"
        f"hqdn3d=2.5:1.5:3.0:3.0,"
        f"deflicker=mode=pm:size={deflicker_size},"
        f"eq=contrast=1.15:brightness=0.02,"
        f"scale=iw:ih,"
        f"unsharp=3:3:0.7:3:3:0.0"
    )

    # --------------------------
    # Step 1: FFmpeg restoration
    # --------------------------
    restored_temp = Path(output_path).with_suffix(".intermediate.mp4")

    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-vf", vf_filters,
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "22",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-ar", "44100",
        "-movflags", "+faststart",
        str(restored_temp),
    ]

    print(f"[INFO] Running FFmpeg restoration: {' '.join(cmd)}")

    process = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    time_pattern = re.compile(r'time=(\d+):(\d+):(\d+).(\d+)')

    progress_bar = tqdm(total=total_frames, desc="FFmpeg Restore", unit="frame", unit_scale=True, smoothing=0.1)
    last_frame = 0

    for line in process.stderr:
        match = time_pattern.search(line)
        if match:
            h, m, s, ms = map(int, match.groups())
            seconds = timestamp_to_seconds(h, m, s, ms)
            current_frame = int(fps * seconds)
            if current_frame > last_frame:
                progress_bar.update(current_frame - last_frame)
                last_frame = current_frame

    process.wait()
    progress_bar.update(total_frames - last_frame)
    progress_bar.close()

    if process.returncode != 0:
        print("[ERROR] FFmpeg failed during restoration.")
        return

    restored_frame_count = get_frame_count(restored_temp)
    print(f"[INFO] Intermediate restored frame count: {restored_frame_count}")

    if restored_frame_count < 5:
        print("[WARN] Restored output seems empty — copying original instead.")
        shutil.copy2(input_path, output_path)
        return

    # --------------------------
    # Step 2: Run Bringing-Old-Photos-Back-to-Life
    # --------------------------
    print(f"[INFO] Running Bringing-Old-Photos-Back-to-Life video restoration...")

    run_video_cmd = [
        "python", "run_video.py",
        "--input_video", str(restored_temp),
        "--output_video", str(output_path),
        "--GPU", str(gpu_id),
        "--with_scratch",
        "--HR"
    ]

    print(f"[INFO] Executing: {' '.join(run_video_cmd)}")

    ret = subprocess.call(run_video_cmd)
    if ret != 0:
        print("[ERROR] Restoration pipeline failed.")
        return
    else:
        print(f"[INFO] Restoration completed successfully. Final video saved at: {output_path}")

    # --------------------------
    # Step 3: Cleanup
    # --------------------------
    try:
        restored_temp.unlink()
        print(f"[INFO] Removed intermediate file: {restored_temp}")
    except Exception:
        pass





def restore_bw_film_cached(input_path, output_path):
    """
    Restore B&W film, using filecache to avoid reprocessing.
    - input_path: current input (actual file to process)
    """

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
    print(f"[DONE] Restored video created at: {output_path}")
    return output_path

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Restore a black & white video using FFmpeg filters.")
    parser.add_argument("input", help="Path to the input video file (B&W)")
    parser.add_argument("output", help="Path to save the restored video")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input file does not exist: {args.input}")
        exit(1)

    try:
        restore_bw_film_cached(args.input, args.output)
    except Exception as e:
        print(f"[ERROR] Restoration failed: {e}")
        exit(1)

