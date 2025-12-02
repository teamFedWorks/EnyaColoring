from scenedetect import detect, AdaptiveDetector, split_video_ffmpeg
import os
import cv2
import numpy as np
import subprocess
import re
from tqdm import tqdm 
from pathlib import Path
import sys
from Utils.main_utils import apply_clahe_in_place, background_upscale_video_onnx_cached
import shutil
import torch, gc
# from scenedetect.scene_manager import SceneManager
# from scenedetect.video_manager import VideoManager

def clear_gpu():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()
    
def extract_best_frame_in_second(video_path, output_path, start_time_sec, duration_sec=1.0, fps=30):
    """
    Extracts the most representative (thumbnail-style) frame from a 1-second interval
    using FFmpeg's native thumbnail=N filter logic, with automatic adaptive deinterlacing.
    """
    frame_window = int(duration_sec * fps)  # e.g., 1s * 30fps = 30 frames

    # print(f"🔍 Checking if '{video_path}' is interlaced...")
    # deinterlace_needed = is_interlaced_ffmpeg(video_path)

    # if deinterlace_needed:
    #     print("📺 Detected interlaced video → applying 'bwdif'")
    #     vf_filter = f"bwdif,thumbnail={frame_window}"
    # else:
    #     print("✅ Progressive video → no deinterlacing applied")
    #     vf_filter = f"thumbnail={frame_window}"
    vf_filter = f"thumbnail={frame_window}"
    # FFmpeg command: seek, filter, and extract best frame
    cmd = [
        "ffmpeg",
        "-ss", str(start_time_sec),
        "-t", str(duration_sec),
        "-i", video_path,
        "-vf", vf_filter,
        "-frames:v", "1",
        "-y",
        output_path
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"🖼️ Saved best frame from {start_time_sec}s to: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ FFmpeg failed to extract frame: {e}")

def darkness_skew(gray_frame, threshold=55):
    dark_pixels = np.sum(gray_frame < threshold)
    total_pixels = gray_frame.size
    return dark_pixels / total_pixels





import cv2
import numpy as np
import subprocess
from pathlib import Path
from typing import Optional

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _ffmpeg_extract_clip(src: str, dst: str, start: float, duration: float) -> bool:
    """Extract subclip from video."""
    try:
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-y", "-ss", str(start), "-t", str(duration),
            "-i", src, "-c", "copy", dst
        ]
        subprocess.run(cmd, check=True)
        return True
    except Exception:
        return False


def _ffmpeg_thumbnail(video_path: str, output_path: str, fps: int = 30, duration_sec: int = 1) -> bool:
    """Use ffmpeg thumbnail filter to pick a representative frame."""
    frame_window = int(duration_sec * fps)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-y", "-i", video_path,
        "-vf", f"thumbnail={frame_window}",
        "-frames:v", "1",
        output_path
    ]
    try:
        subprocess.run(cmd, check=True)
        return True
    except Exception:
        return False


def _sharpness_score(frame: np.ndarray) -> float:
    """Compute Laplacian variance sharpness score."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
#yolo
def extract_best_frame_in_second_(video_path: str,
                                 output_path: str,
                                 start_time_sec: float,
                                 duration_sec: float = 1.0,
                                 fps: int = 30,
                                 yolo_weights: str = "yolo11n.pt",
                                 fallback: str = "thumbnail",
                                 min_conf: float = 0.6) -> Optional[int]:
    """
    Extract best frame from a time interval in a video:
    - If persons detected with confidence >= min_conf → frame with max person area
    - Else → fallback ('thumbnail' or 'sharpness')

    Args:
        video_path: input video
        output_path: path to save best frame
        start_time_sec: interval start in seconds
        duration_sec: interval length in seconds
        fps: frames per second (used for thumbnail fallback)
        yolo_weights: YOLO model weights
        fallback: "thumbnail" or "sharpness"
        min_conf: minimum confidence threshold for person detection

    Returns:
        Absolute frame index in original video if person-based chosen, else None
    """
    # Extract subclip for interval
    tmp_clip = str(Path(output_path).with_suffix(".clip.mp4"))
    if not _ffmpeg_extract_clip(video_path, tmp_clip, start_time_sec, duration_sec):
        print("[extract_best_frame_in_second] Failed to extract subclip")
        return None

    # Load YOLO
    try:
        from ultralytics import YOLO
        model = YOLO(yolo_weights)
    except Exception as e:
        print(f"[extract_best_frame_in_second] YOLO failed to load: {e}")
        model = None

    cap = cv2.VideoCapture(tmp_clip)
    if not cap.isOpened():
        return None

    best_idx, best_area = None, -1
    best_frame = None

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if model:
            results = model(frame, verbose=False)
            for r in results:
                if not hasattr(r, "boxes") or r.boxes is None:
                    continue
                for box in r.boxes:
                    if int(box.cls) != 0:  # only class 0 = person
                        continue
                    if float(box.conf) < min_conf:  # confidence filter
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    area = max(0, (x2 - x1) * (y2 - y1))
                    if area > best_area:
                        best_area = area
                        best_idx = frame_idx
                        best_frame = frame.copy()
        frame_idx += 1

    cap.release()

    # Case 1: Found confident person → save and return absolute frame index
    if best_frame is not None:
        cv2.imwrite(output_path, best_frame)
        abs_idx = int(start_time_sec * fps) + best_idx
        return abs_idx

    # Case 2: Fallback
    if fallback == "sharpness":
        cap = cv2.VideoCapture(tmp_clip)
        best_score, best_frame, best_idx = -1.0, None, None
        fi = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            score = _sharpness_score(frame)
            if score > best_score:
                best_score, best_frame, best_idx = score, frame.copy(), fi
            fi += 1
        cap.release()
        if best_frame is not None:
            cv2.imwrite(output_path, best_frame)
        return None
    else:  # fallback == "thumbnail"
        _ffmpeg_thumbnail(tmp_clip, output_path, fps=fps, duration_sec=duration_sec)
        return None







from scenedetect.frame_timecode import FrameTimecode


def split_scenes_max_duration(scene_list, video_fps, max_duration_sec=10):
    """
    Splits scenes from scene_list into sub-scenes no longer than max_duration_sec.

    Args:
        scene_list (List[Tuple[FrameTimecode, FrameTimecode]]): List of original scenes.
        video_fps (float): Frames per second of the video.
        max_duration_sec (int or float): Maximum allowed duration per scene.

    Returns:
        List[Tuple[FrameTimecode, FrameTimecode]]: Updated scene list with enforced max duration.
    """
    final_scenes = []

    for start_time, end_time in scene_list:
        start_sec = start_time.get_seconds()
        end_sec = end_time.get_seconds()
        current = start_sec

        while current < end_sec:
            next_cut = min(current + max_duration_sec, end_sec)
            start_tc = FrameTimecode(current, video_fps)
            end_tc = FrameTimecode(next_cut, video_fps)
            final_scenes.append((start_tc, end_tc))
            current = next_cut

    return final_scenes


def split_scenes_max_duration(scene_list, video_fps, max_duration_sec=10):
    """
    Splits scenes into sub-scenes no longer than max_duration_sec.
    Skips zero-length scenes (same start and end) only in the final returned list.
    """
    from scenedetect.frame_timecode import FrameTimecode

    final_scenes = []

    for start_time, end_time in scene_list:
        start_sec = start_time.get_seconds()
        end_sec = end_time.get_seconds()

        current = start_sec
        while current < end_sec:
            next_cut = min(current + max_duration_sec, end_sec)
            start_tc = FrameTimecode(current, video_fps)
            end_tc = FrameTimecode(next_cut, video_fps)
            final_scenes.append((start_tc, end_tc))
            current = next_cut

    # ✅ Filter zero-length scenes only in final output
    cleaned_scenes = [
        (s, e)
        for (s, e) in final_scenes
        if (e.get_seconds() - s.get_seconds()) != 0 and (e.get_frames() - s.get_frames()) != 0
    ]

    if len(cleaned_scenes) < len(final_scenes):
        print(f"[⚠️] Removed {len(final_scenes) - len(cleaned_scenes)} zero-length scenes.")

    print(f"[✅] Returning {len(cleaned_scenes)} valid scenes.")
    return cleaned_scenes






    
def split_video_by_scene(input_video_path, output_dir, original_input_path, scale=2, fps=30):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Detecting scenes in: {input_video_path}")
    #scene_list = detect(input_video_path, AdaptiveDetector(min_scene_len=1, luma_only=True))
    #scene_list = detect(input_video_path, AdaptiveDetector(adaptive_threshold =1.5, min_scene_len=1, luma_only=True, window_width = 2, min_content_val = 5)) 
    scene_list = detect(input_video_path, AdaptiveDetector(adaptive_threshold =2.5, min_scene_len=1, luma_only=True, window_width = 2)) 
    print(f"Detected {len(scene_list)} scenes.")
    
    #print(scene_list)
    # === Fallback logic: treat full video as one scene if no scene is found
    from scenedetect.frame_timecode import FrameTimecode
    
    if len(scene_list) == 0:
        cap = cv2.VideoCapture(input_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    
        print("⚠️ No scenes detected — using entire video as one scene.")
        scene_list = [(FrameTimecode(0, fps), FrameTimecode(total_frames, fps))]
    #scene_list = split_scenes_max_duration(scene_list, fps, max_duration_sec=2)
    split_video_ffmpeg(
        input_video_path=input_video_path,
        scene_list=scene_list,
        output_dir=output_dir,
        show_progress=True,
        show_output=True,
    )

    video_name = os.path.splitext(os.path.basename(input_video_path))[0]
    image_output_dir = os.path.join(output_dir, f"{video_name}_images")
    os.makedirs(image_output_dir, exist_ok=True)

    scene_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".mp4")])
    for filename in tqdm(scene_files, desc="Extracting scene frames", unit="scene"):

        scene_path = os.path.join(output_dir, filename)
        scene_name = os.path.splitext(filename)[0]

        if(False):
             # === Apply CLAHE first based on brightness analysis ===
            intensity_thresh = 55
            skew_thresh = 0.6
            cap = cv2.VideoCapture(scene_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            intensity_vals = []
            skew_vals = []
    
            for _ in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                intensity_vals.append(np.mean(gray))
                skew_vals.append(darkness_skew(gray, threshold=intensity_thresh))
            cap.release()
    
            mean_intensity = np.mean(intensity_vals)
            mean_skew = np.mean(skew_vals)
    
            print(f"🧪 Scene: {scene_name} | Mean: {mean_intensity:.2f}, Skew: {mean_skew:.2f}")
            if mean_intensity <= intensity_thresh and mean_skew >= skew_thresh:
                print(f"✨ CLAHE applied to: {scene_path}")
                apply_clahe_in_place(scene_path)
            else:
                print(f"✅ CLAHE skipped: Good brightness for scene {scene_name}")
            background_upscaled_video_path = background_upscale_video_onnx_cached(scene_path, original_input_path, scale)
            shutil.move(background_upscaled_video_path, scene_path)
            upscale_dir = os.path.dirname(background_upscaled_video_path) 
            # Clean up: delete the folder if it's now empty
            try:
                shutil.rmtree(upscale_dir)
                print(f"🧹 Deleted temporary folder: {upscale_dir}")
            except Exception as e:
                print(f"⚠️ Failed to delete folder {upscale_dir}: {e}")
            print(f"✅  background upscaled for scene : {scene_name}")
            clear_gpu()

        cap = cv2.VideoCapture(scene_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        scene_fps = cap.get(cv2.CAP_PROP_FPS) or fps
        duration = total_frames / scene_fps
        cap.release()

        # First frame (_1)
        first_frame_path = os.path.join(image_output_dir, f"{scene_name}_1.jpg")
        if duration > 4:
            extract_best_frame_in_second(scene_path, first_frame_path, start_time_sec=1.0, fps=int(scene_fps))
        else:
            extract_best_frame_in_second(scene_path, first_frame_path, start_time_sec=0.0, fps=int(scene_fps))

        # Middle frame (_2)
        middle_frame_path = os.path.join(image_output_dir, f"{scene_name}_2.jpg")
        if duration > 4:
            middle_start = max((duration / 2.0) - 0.5, 0)
            extract_best_frame_in_second(scene_path, middle_frame_path, start_time_sec=middle_start, fps=int(scene_fps))
        else:
            cap = cv2.VideoCapture(scene_path)
            if cap.isOpened():
                middle_frame_num = total_frames // 2
                cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_num)
                ret, frame = cap.read()
                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(middle_frame_path, gray)
                    print(f"Scene {scene_name} short video, extracted middle frame to: {middle_frame_path}")
                else:
                    print(f"Failed to read middle frame from: {scene_path}")
                cap.release()

        # Last frame (_3)
        last_frame_path = os.path.join(image_output_dir, f"{scene_name}_3.jpg")
        if duration > 4:
            extract_best_frame_in_second(scene_path, last_frame_path, start_time_sec=duration - 1.0, fps=int(scene_fps))
        else:
            cap = cv2.VideoCapture(scene_path)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
                ret, frame = cap.read()
                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(last_frame_path, gray)
                    print(f"Scene {scene_name} short video, extracted last frame to: {last_frame_path}")
                else:
                    print(f"Failed to read last frame from: {scene_path}")
                cap.release()


#split_video_by_scene_m
def split_video_by_scene(input_video_path, output_dir, original_input_path, scale=2, fps=30):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Detecting scenes in: {input_video_path}")
    scene_list = detect(input_video_path, AdaptiveDetector(adaptive_threshold=2.5, min_scene_len=1, luma_only=True, window_width=2)) 
    print(f"Detected {len(scene_list)} scenes.")
    
    from scenedetect.frame_timecode import FrameTimecode
    
    if len(scene_list) == 0:
        cap = cv2.VideoCapture(input_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    
        print("⚠️ No scenes detected — using entire video as one scene.")
        scene_list = [(FrameTimecode(0, fps), FrameTimecode(total_frames, fps))]
    print("previous", scene_list)
    scene_list = split_scenes_max_duration(scene_list, fps, max_duration_sec=1)
    print("current", scene_list)
    split_video_ffmpeg(
        input_video_path=input_video_path,
        scene_list=scene_list,
        output_dir=output_dir,
        show_progress=True,
        show_output=True,
    )

    video_name = os.path.splitext(os.path.basename(input_video_path))[0]
    image_output_dir = os.path.join(output_dir, f"{video_name}_images")
    os.makedirs(image_output_dir, exist_ok=True)

    scene_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".mp4")])
    for filename in tqdm(scene_files, desc="Extracting middle frames", unit="scene"):

        scene_path = os.path.join(output_dir, filename)
        scene_name = os.path.splitext(filename)[0]

        cap = cv2.VideoCapture(scene_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        scene_fps = cap.get(cv2.CAP_PROP_FPS) or fps
        duration = total_frames / scene_fps
        cap.release()

        # Middle frame (_m)
        middle_frame_path = os.path.join(image_output_dir, f"{scene_name}_m.jpg")
        if duration > 4:
            middle_start = max((duration / 2.0) - 0.5, 0)
            extract_best_frame_in_second(scene_path, middle_frame_path, start_time_sec=middle_start, fps=int(scene_fps))
        else:
            cap = cv2.VideoCapture(scene_path)
            if cap.isOpened():
                middle_frame_num = total_frames // 2
                cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_num)
                ret, frame = cap.read()
                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(middle_frame_path, gray)
                    print(f"Scene {scene_name} short video, extracted middle frame to: {middle_frame_path}")
                else:
                    print(f"Failed to read middle frame from: {scene_path}")
                cap.release()







def split_video_by_scene(input_video_path, output_dir, original_input_path, scale=2, fps=30):
    import shutil, re
    os.makedirs(output_dir, exist_ok=True)

    print(f"Detecting scenes in: {input_video_path}")
    scene_list = detect(input_video_path, AdaptiveDetector(adaptive_threshold=2.5, min_scene_len=1, luma_only=True, window_width=2)) 
    print(f"Detected {len(scene_list)} scenes. (Previous)")

    from scenedetect.frame_timecode import FrameTimecode
    
    if len(scene_list) == 0:
        cap = cv2.VideoCapture(input_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        print("⚠️ No scenes detected — using entire video as one scene.")
        scene_list = [(FrameTimecode(0, fps), FrameTimecode(total_frames, fps))]

    prev_scene_list = scene_list[:]  # keep natural scenes
    print("previous", prev_scene_list)

    # split into 1 sec sub-scenes
    scene_list = split_scenes_max_duration(scene_list, fps, max_duration_sec=1)
    print("current", scene_list)

    split_video_ffmpeg(
        input_video_path=input_video_path,
        scene_list=scene_list,
        output_dir=output_dir,
        show_progress=True,
        show_output=True,
    )

    video_name = os.path.splitext(os.path.basename(input_video_path))[0]

    # existing folder (unchanged)
    image_output_dir = os.path.join(output_dir, f"{video_name}_images")
    os.makedirs(image_output_dir, exist_ok=True)

    # new folder (renamed copies)
    renamed_output_dir = os.path.join(output_dir, f"{video_name}_images_prev")
    os.makedirs(renamed_output_dir, exist_ok=True)

    scene_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".mp4")])

    # map current sub-scenes → previous natural scene index
    curr_to_prev = {}
    for c_idx, (c_start, c_end) in enumerate(scene_list, start=1):
        for p_idx, (p_start, p_end) in enumerate(prev_scene_list, start=1):
            if c_start.get_seconds() >= p_start.get_seconds() and c_end.get_seconds() <= p_end.get_seconds():
                curr_to_prev[c_idx] = p_idx
                break

    # Step 1: Extract frames and save renamed copies
    for idx, filename in enumerate(tqdm(scene_files, desc="Extracting middle frames", unit="scene"), start=1):
        scene_path = os.path.join(output_dir, filename)
        scene_name = os.path.splitext(filename)[0]

        cap = cv2.VideoCapture(scene_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        scene_fps = cap.get(cv2.CAP_PROP_FPS) or fps
        duration = total_frames / scene_fps
        cap.release()

        middle_frame_path = os.path.join(image_output_dir, f"{scene_name}_m.jpg")

        if duration > 4:
            middle_start = max((duration / 2.0) - 0.5, 0)
            extract_best_frame_in_second(scene_path, middle_frame_path, start_time_sec=middle_start, fps=int(scene_fps))
        else:
            cap = cv2.VideoCapture(scene_path)
            if cap.isOpened():
                middle_frame_num = total_frames // 2
                cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_num)
                ret, frame = cap.read()
                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(middle_frame_path, gray)
                    print(f"Scene {scene_name} short video, extracted middle frame to: {middle_frame_path}")
                cap.release()

        # Copy into renamed folder with PrevScene suffix
        prev_scene_num = curr_to_prev.get(idx, 0)
        renamed_frame_path = os.path.join(
            renamed_output_dir,
            f"{scene_name}_m_(PrevScene{prev_scene_num}).jpg"
        )
        shutil.copy2(middle_frame_path, renamed_frame_path)

    print(f"\n✅ Frames saved in:\n- {image_output_dir} (original)\n- {renamed_output_dir} (with PrevScene#)")

    # Step 2: Group by PrevScene
    pattern = re.compile(r"_\(PrevScene(\d+)\)")
    scene_groups = {}
    for fname in sorted(os.listdir(renamed_output_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        match = pattern.search(fname)
        if not match:
            continue
        prev_scene_num = int(match.group(1))
        scene_groups.setdefault(prev_scene_num, []).append(fname)

    # Step 3: Create a combined "_prevscene.mp4" (one frame per PrevScene)
    selected_frames = []
    for scene_num in sorted(scene_groups.keys()):
        frames = sorted(scene_groups[scene_num])
        n = len(frames)
        if n == 0: 
            continue
        if n % 2 == 1:  # odd
            mid_idx = n // 2
        else:          # even → take lower middle
            mid_idx = (n // 2) - 1
        selected_frames.append(os.path.join(renamed_output_dir, frames[mid_idx]))

    if selected_frames:
        first_frame = cv2.imread(selected_frames[0])
        height, width = first_frame.shape[:2]
        prevscene_video_path = os.path.join(renamed_output_dir, f"{video_name}_prevscene.mp4")
        out = cv2.VideoWriter(prevscene_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        for f in selected_frames:
            img = cv2.imread(f)
            if img is not None:
                out.write(img)
        out.release()
        print(f"🎥 Created combined video: {prevscene_video_path}")

    # Step 4: Create individual videos for each PrevScene outside renamed_output_dir
    for scene_num, frames in scene_groups.items():
        frames = sorted(frames)
        first_frame = cv2.imread(os.path.join(renamed_output_dir, frames[0]))
        height, width = first_frame.shape[:2]
        indiv_path = os.path.join(output_dir, f"PrevScene{scene_num}.mp4")
        out = cv2.VideoWriter(indiv_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        for f in frames:
            img = cv2.imread(os.path.join(renamed_output_dir, f))
            if img is not None:
                out.write(img)
        out.release()
        print(f"🎬 Created individual video: {indiv_path}")



def split_video_by_scene(input_video_path, output_dir, original_input_path, scale=2, fps=30):
    print_flag = False
    import shutil, re
    import cv2, os, json
    from tqdm import tqdm
    os.makedirs(output_dir, exist_ok=True)
    if(print_flag):
        print(f"Detecting scenes in: {input_video_path}")
    scene_list = detect(input_video_path, AdaptiveDetector(adaptive_threshold=2.5, min_scene_len=1, luma_only=True, window_width=2)) 
    print(f"Detected {len(scene_list)} scenes. (Previous)")

    from scenedetect.frame_timecode import FrameTimecode
    
    if len(scene_list) == 0:
        cap = cv2.VideoCapture(input_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        print("⚠️ No scenes detected — using entire video as one scene.")
        scene_list = [(FrameTimecode(0, fps), FrameTimecode(total_frames, fps))]

    prev_scene_list = scene_list[:]  # keep natural scenes
    if(print_flag):
       print("previous", prev_scene_list)

    # split into 1 sec sub-scenes
    scene_list = split_scenes_max_duration(scene_list, fps, max_duration_sec=1)
    if(print_flag):
       print("current", scene_list)

    split_video_ffmpeg(
        input_video_path=input_video_path,
        scene_list=scene_list,
        output_dir=output_dir,
        show_progress=True,
        show_output=True,
    )

    video_name = os.path.splitext(os.path.basename(input_video_path))[0]

    # existing folder (unchanged)
    image_output_dir = os.path.join(output_dir, f"{video_name}_images")
    os.makedirs(image_output_dir, exist_ok=True)

    # new folder (renamed copies)
    renamed_output_dir = os.path.join(output_dir, f"{video_name}_images_prev")
    os.makedirs(renamed_output_dir, exist_ok=True)

    scene_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".mp4")])

    # map current sub-scenes → previous natural scene index
    curr_to_prev = {}
    for c_idx, (c_start, c_end) in enumerate(scene_list, start=1):
        for p_idx, (p_start, p_end) in enumerate(prev_scene_list, start=1):
            if c_start.get_seconds() >= p_start.get_seconds() and c_end.get_seconds() <= p_end.get_seconds():
                curr_to_prev[c_idx] = p_idx
                break

    # Step 1: Extract frames and save renamed copies, also store original frame numbers
    frame_number_map = {}  # prev_scene_num -> list of original frame numbers

    for idx, filename in enumerate(tqdm(scene_files, desc="Extracting middle frames", unit="scene"), start=1):
        scene_path = os.path.join(output_dir, filename)
        scene_name = os.path.splitext(filename)[0]

        cap = cv2.VideoCapture(scene_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        scene_fps = cap.get(cv2.CAP_PROP_FPS) or fps
        duration = total_frames / scene_fps
        cap.release()

        middle_frame_path = os.path.join(image_output_dir, f"{scene_name}_m.jpg")

        if duration > 4:
            middle_start = max((duration / 2.0) - 0.5, 0)
            mid_frame = extract_best_frame_in_second(scene_path, middle_frame_path, start_time_sec=middle_start, fps=int(scene_fps))
        else:
            cap = cv2.VideoCapture(scene_path)
            if cap.isOpened():
                middle_frame_num = total_frames // 2
                cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_num)
                ret, frame = cap.read()
                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(middle_frame_path, gray)
                    mid_frame = int(scene_list[idx-1][0].get_frames()) + middle_frame_num  # original frame number
                cap.release()

        # Copy into renamed folder with PrevScene suffix
        prev_scene_num = curr_to_prev.get(idx, 0)
        renamed_frame_path = os.path.join(
            renamed_output_dir,
            f"{scene_name}_m_(PrevScene{prev_scene_num}).jpg"
        )
        shutil.copy2(middle_frame_path, renamed_frame_path)

        # Store frame number for JSON
        frame_number_map.setdefault(prev_scene_num, []).append(mid_frame)

    if(print_flag):
       print(f"\n✅ Frames saved in:\n- {image_output_dir} (original)\n- {renamed_output_dir} (with PrevScene#)")

    # Step 2: Group by PrevScene
    pattern = re.compile(r"_\(PrevScene(\d+)\)")
    scene_groups = {}
    for fname in sorted(os.listdir(renamed_output_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        match = pattern.search(fname)
        if not match:
            continue
        prev_scene_num = int(match.group(1))
        scene_groups.setdefault(prev_scene_num, []).append(fname)

    # Step 3: Create a combined "_prevscene.mp4" (one frame per PrevScene)
    selected_frames = []
    for scene_num in sorted(scene_groups.keys()):
        frames = sorted(scene_groups[scene_num])
        n = len(frames)
        if n == 0: 
            continue
        if n % 2 == 1:  # odd
            mid_idx = n // 2
        else:          # even → take lower middle
            mid_idx = (n // 2) - 1
        selected_frames.append(os.path.join(renamed_output_dir, frames[mid_idx]))

    if selected_frames:
        first_frame = cv2.imread(selected_frames[0])
        height, width = first_frame.shape[:2]
        prevscene_video_path = os.path.join(renamed_output_dir, f"{video_name}_prevscene.mp4")
        out = cv2.VideoWriter(prevscene_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        for f in selected_frames:
            img = cv2.imread(f)
            if img is not None:
                out.write(img)
        out.release()
        if(print_flag):
           print(f"🎥 Created combined video: {prevscene_video_path}")

    # Step 4: Create individual videos for each PrevScene outside renamed_output_dir
   # for scene_num, frames in scene_groups.items():
    from tqdm import tqdm
    
    # High-level tqdm for scenes
    for scene_num, frames in tqdm(scene_groups.items(), desc="Scenes", unit="scene"):
        frames = sorted(frames)
        first_frame = cv2.imread(os.path.join(renamed_output_dir, frames[0]))
        height, width = first_frame.shape[:2]
        indiv_path = os.path.join(output_dir, f"PrevScene{scene_num}.mp4")
        out = cv2.VideoWriter(indiv_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        for f in frames:
            img = cv2.imread(os.path.join(renamed_output_dir, f))
            if img is not None:
                out.write(img)
        out.release()
        if(print_flag):
          print(f"🎬 Created individual video: {indiv_path}")

    # Step 5: Write JSON files containing original frame numbers
    # for scene_num, frames in frame_number_map.items():
    #     json_path = os.path.join(output_dir, f"PrevScene{scene_num}_frames.json")
    #     with open(json_path, "w", encoding="utf-8") as jf:
    #         json.dump(frames, jf)
    #     if(print_flag):
    #        print(f"📄 Saved frame numbers for PrevScene{scene_num} → {json_path}")

    
    # # Step 6: Create full original PrevScene videos (entire natural scenes)
    # print("\n🎞 Creating FULL PrevScene videos from original input...")

    # cap = cv2.VideoCapture(original_input_path)
    # orig_fps = cap.get(cv2.CAP_PROP_FPS) or fps

    # for idx, (start_ft, end_ft) in enumerate(prev_scene_list, start=1):
    #     start_frame = start_ft.get_frames()
    #     end_frame = end_ft.get_frames()
    #     total_scene_frames = end_frame - start_frame

    #     # Output path
    #     full_out_path = os.path.join(output_dir, f"PrevScene{idx}_entire.mp4")

    #     # Move to start frame
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    #     # Read first frame to get size
    #     ret, frame = cap.read()
    #     if not ret:
    #         print(f"⚠️ Could not read start frame for PrevScene{idx}")
    #         continue

    #     height, width = frame.shape[:2]
    #     out = cv2.VideoWriter(full_out_path, cv2.VideoWriter_fourcc(*'mp4v'), orig_fps, (width, height))

    #     # Write first frame
    #     out.write(frame)

    #     # Write remaining frames
    #     for f in range(start_frame + 1, end_frame):
    #         ret, frame = cap.read()
    #         if not ret:
    #             break
    #         out.write(frame)

    #     out.release()
    #     print(f"🎬 Created FULL original scene: {full_out_path}")

    # cap.release()





def create_video_from_grayscale_frames(frame_dir, output_video_name="preview.mp4", fps=30):
    frame_files = sorted([
        f for f in os.listdir(frame_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if not frame_files:
        print("No frames found to create video.")
        return

    first_frame = cv2.imread(os.path.join(frame_dir, frame_files[0]), cv2.IMREAD_GRAYSCALE)
    height, width = first_frame.shape

    output_video_path = os.path.join(frame_dir, output_video_name)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), isColor=False)

    print(f"🎥 Writing {len(frame_files)} frames to preview: {output_video_path}")
    for fname in tqdm(frame_files, desc="Generating video", unit="frame"):
        img = cv2.imread(os.path.join(frame_dir, fname), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            out.write(img)

    out.release()
    print("✅ Scene refs video saved.")
    return output_video_path



def run_scene_split_cached(input_path: str,  output_dir: str) -> str:
    from Utils.pyscene_split_utils import split_video_by_scene, create_video_from_grayscale_frames
    """
    Run scene detection + scene_refs video generation with folder hashing.
    Returns full path to the scene_refs video.
    """
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
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
        split_video_by_scene(input_path, str(output_dir), fps=fps)
    
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
    
    print(f"[DONE] Scene-split refs video saved at: {scene_refs_bw_path}")
    return str(scene_refs_bw_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="scene detection + scene_refs video generation.")
    parser.add_argument("input", help="Path to the input video file")
    parser.add_argument("output", help="Ouput directory path")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input file does not exist: {args.input}")
        exit(1)

    try:
        run_scene_split_cached(args.input, args.output)
    except Exception as e:
        print(f"[ERROR] scene detection + scene_refs: {e}")
        exit(1)
