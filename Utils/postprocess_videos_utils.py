import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os

# === Parameters ===
MAX_FRAMES = 300
WARM_THRESH_A = 132
WARM_THRESH_B = 135
COOL_STRENGTH_A = 5
COOL_STRENGTH_B = 10
SAT_FACTOR = 1.3
VAL_FACTOR = 1.1
CONTRAST_FACTOR = 1.2
BRIGHTNESS_OFFSET = 10

def enhance_frame(frame):
    # === Convert to LAB and check warmth ===
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    mean_a, mean_b = np.mean(a), np.mean(b)

    # === Apply cooling if warm ===
   # if mean_a > WARM_THRESH_A or mean_b > WARM_THRESH_B:
    a = np.clip(a.astype(np.int16) - COOL_STRENGTH_A, 0, 255).astype(np.uint8)
    b = np.clip(b.astype(np.int16) - COOL_STRENGTH_B, 0, 255).astype(np.uint8)
    lab = cv2.merge((l, a, b))
    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # === Enhance saturation and brightness ===
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * SAT_FACTOR, 0, 255)
    v = np.clip(v * VAL_FACTOR, 0, 255)
    hsv_enhanced = cv2.merge((h, s, v)).astype(np.uint8)
    frame = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

    # === Final contrast and brightness adjustment ===
    frame_final = cv2.convertScaleAbs(frame, alpha=CONTRAST_FACTOR, beta=BRIGHTNESS_OFFSET)

    return frame_final

def process_video(input_path: str, output_path: str):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(min(MAX_FRAMES, cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"[INFO] Processing up to {total_frames} frames at {fps:.2f} FPS: {width}x{height}")
    pbar = tqdm(total=total_frames, desc="✨ Enhancing video")

    frame_count = 0
    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = enhance_frame(frame)
        out_writer.write(processed_frame)
        frame_count += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    out_writer.release()
    print(f"[✅] Saved enhanced video → {output_path}")




def enhance_saturation_and_brightness(
    video_path, 
    output_path, 
    saturation_factor=1.1, 
    value_factor=1.1, 
    max_saturation=255, 
    max_value=255
):
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Open input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"❌ Cannot open video: {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define output writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process each frame
    with tqdm(total=total_frames, desc="Enhancing HSV", unit="frame") as pbar:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            # Convert to HSV
            frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

            # Enhance saturation and value with clamping to custom maximums
            frame_hsv[..., 1] = np.minimum(frame_hsv[..., 1] * saturation_factor, max_saturation)  # Saturation
            frame_hsv[..., 2] = np.minimum(frame_hsv[..., 2] * value_factor, max_value)            # Brightness (Value)

            # Convert back to uint8
            frame_hsv = frame_hsv.astype(np.uint8)

            # Convert back to BGR and write frame
            frame_enhanced = cv2.cvtColor(frame_hsv, cv2.COLOR_HSV2BGR)
            out.write(frame_enhanced)
            pbar.update(1)

    cap.release()
    out.release()
    print(f"✅ Enhanced video saved to: {output_path}")



def postprocess_videos_cached(input_path, output_path):

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
    finally:
        import torch, gc
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
    print(f"[DONE] postprocess video created at: {output_path}")
    return output_path

    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Postprocess a video to boost color and brightness.")
    parser.add_argument("input", help="Path to the input colorized video")
    parser.add_argument("output", help="Output path")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input file does not exist: {args.input}")
        exit(1)

    try:
        final_output = postprocess_videos_cached(args.input, args.output)
        print(f"✔️ Postprocessing completed. Final video at: {final_output}")
    except Exception as e:
        print(f"[ERROR] Postprocessing failed: {e}")
        exit(1)


