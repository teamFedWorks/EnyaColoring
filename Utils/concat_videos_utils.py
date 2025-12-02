import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm  # ✅ Added

def get_video_properties(cap):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return width, height, fps, total_frames

def resize_video_frame(frame, scale_factor):
    h, w = frame.shape[:2]
    return cv2.resize(frame, (int(w * scale_factor), int(h * scale_factor)), interpolation=cv2.INTER_AREA)

def concat_video_frames(
    original_video_path: str,
    colorized_video_path: str,
    output_video_path: str
):
    cap_orig = cv2.VideoCapture(original_video_path)
    cap_color = cv2.VideoCapture(colorized_video_path)

    if not cap_orig.isOpened():
        raise ValueError(f"Cannot open original video: {original_video_path}")
    if not cap_color.isOpened():
        raise ValueError(f"Cannot open colorized video: {colorized_video_path}")

    color_width, color_height, fps, total_frames = get_video_properties(cap_color)
    layout = "landscape" if color_width > color_height else "portrait"
    scale_factor = 1 / 3.0

    orig_frame_sample = resize_video_frame(cap_orig.read()[1], scale_factor)
    orig_resized_h, orig_resized_w = orig_frame_sample.shape[:2]

    if layout == "landscape":
        out_width = max(color_width, orig_resized_w)
        out_height = color_height + orig_resized_h
    else:
        out_width = color_width + orig_resized_w
        out_height = max(color_height, orig_resized_h)

    print(f"[INFO] Layout: {layout} | Output: {out_width}x{out_height}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (out_width, out_height))

    pbar = tqdm(total=total_frames, desc="🔄 Writing frames")

    while True:
        ret_orig, frame_orig = cap_orig.read()
        ret_color, frame_color = cap_color.read()
        if not (ret_orig and ret_color):
            break

        frame_orig_resized = resize_video_frame(frame_orig, scale_factor)
        canvas = np.zeros((out_height, out_width, 3), dtype=np.uint8)

        if layout == "landscape":
            canvas[0:orig_resized_h, 0:orig_resized_w] = frame_orig_resized
            canvas[orig_resized_h:orig_resized_h+color_height, 0:color_width] = frame_color
        else:
            canvas[0:orig_resized_h, 0:orig_resized_w] = frame_orig_resized
            canvas[0:color_height, orig_resized_w:orig_resized_w+color_width] = frame_color

        out_writer.write(canvas)
        pbar.update(1)

    pbar.close()
    cap_orig.release()
    cap_color.release()
    out_writer.release()
    print(f"[✅] Saved concatenated video → {output_video_path}")

if __name__ == "__main__":
    original_video_path = "original.mp4"
    colorized_video_path = "colorized.mp4"
    output_video_path = "concatenated_output.mp4"

    concat_video_frames(original_video_path, colorized_video_path, output_video_path)
