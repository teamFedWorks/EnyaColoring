import os
import cv2
import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Using:", torch.cuda.get_device_name(0))
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from deoldify.visualize import get_image_colorizer
from deoldify import device
from deoldify.device_id import DeviceId

# ==== CONFIG ====
YOLO_MODEL_PATH = "models/yolo11x-seg.pt"
CONF_THRESHOLD = 0.6
ENLARGE_ROI = False
ENLARGE_PERCENT = 0.2
# =================

# Setup once
device.set(device=DeviceId.GPU0)
colorizer = get_image_colorizer(artistic=True)
yolo_model = YOLO(YOLO_MODEL_PATH).to("cuda")


def deoldify_inference(frame_rgb):
    """Run DeOldify on an ROI and return colorized numpy array."""
    ret = colorizer.get_transformed_image(frame_rgb, render_factor=16, post_process=True)
    return np.array(ret)


from PIL import Image

def deoldify_inference(frame_rgb):
    """Run DeOldify on an ROI and return colorized numpy array."""
    pil_img = Image.fromarray(frame_rgb).convert("RGB")   # ✅ ensure proper PIL.Image
    ret = colorizer.get_transformed_image(pil_img, render_factor=16, post_process=True)
    return np.array(ret)  # back to numpy



def run_deoldify_yolo(input_path, overwrite=False):
    """
    Run YOLO segmentation + DeOldify colorization on input video.
    Output files are written in the same folder as input:
        <name>_output.mp4  (annotated result)
        <name>_yolo.mp4    (debug video)
    Returns (annotated_output, yolo_debug_output).
    """
    folder, filename = os.path.split(input_path)
    name, _ = os.path.splitext(filename)

    output_path = os.path.join(folder, f"{name}_output.mp4")
    results_frame_output_path = os.path.join(folder, f"{name}_yolo.mp4")

    if os.path.exists(output_path) and not overwrite:
        print(f"[CACHE] Annotated video found: {output_path}")
        return output_path, results_frame_output_path

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"❌ Cannot open input video: {input_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    results_frame_writer = cv2.VideoWriter(results_frame_output_path, fourcc, fps, (width, height))

    try:
        with tqdm(total=total_frames, desc="Enhancing", unit="frame") as pbar:
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                results = yolo_model.predict(frame_bgr, conf=CONF_THRESHOLD, verbose=False)
                results_frame = results[0].plot()
                results_frame_writer.write(results_frame)
                annotated = frame_rgb.copy()

                if results and results[0].boxes is not None and results[0].masks is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    classes = results[0].boxes.cls.cpu().numpy()
                    masks_raw = results[0].masks.data

                    masks_resized = torch.nn.functional.interpolate(
                        masks_raw.unsqueeze(1).float(),
                        size=(height, width),
                        mode="nearest"
                    ).squeeze(1).cpu().numpy()

                    for box, mask, cls in zip(boxes, masks_resized, classes):
                        if int(cls) != 0:
                            continue  # only "person"

                        x1, y1, x2, y2 = map(int, box)

                        if ENLARGE_ROI:
                            box_h = y2 - y1
                            box_w = x2 - x1
                            pad_h = int(ENLARGE_PERCENT * box_h)
                            pad_w = int(ENLARGE_PERCENT * box_w)
                            x1 = max(0, x1 - pad_w)
                            y1 = max(0, y1 - pad_h)
                            x2 = min(width, x2 + pad_w)
                            y2 = min(height, y2 + pad_h)

                        roi = frame_rgb[y1:y2, x1:x2]
                        if roi.size == 0:
                            continue

                        roi_colorized = deoldify_inference(roi)
                        roi_resized = cv2.resize(roi_colorized, (x2 - x1, y2 - y1))

                        mask_crop = mask[y1:y2, x1:x2]
                        if mask_crop.shape != (y2 - y1, x2 - x1):
                            mask_crop = cv2.resize(mask_crop, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
                        mask_bool = mask_crop > 0.5

                        overlay_roi = annotated[y1:y2, x1:x2]
                        overlay_roi[mask_bool] = roi_resized[mask_bool]
                        annotated[y1:y2, x1:x2] = overlay_roi

                out_writer.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                pbar.update(1)

    except (Exception, KeyboardInterrupt) as e:
        print(f"[ERROR] DeOldify YOLO pipeline interrupted: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        raise
    finally:
        cap.release()
        out_writer.release()
        results_frame_writer.release()
        import gc
        torch.cuda.empty_cache()
        gc.collect()

    print(f"[INFO] Annotated video saved: {output_path}")
    print(f"[INFO] YOLO debug video saved: {results_frame_output_path}")
    return output_path, results_frame_output_path



def run_deoldify_yolo_global(input_path, overwrite=False):
    """
    Run DeOldify once per frame for the entire image.
    Then overlay YOLO masks (resized to full frame, no ROI cropping).
    Saves:
        <name>_output_global.mp4  (final annotated with YOLO + DeOldify)
        <name>_deoldify.mp4       (DeOldify-only frames)
        <name>_yolo.mp4           (YOLO debug frames)
    """
    folder, filename = os.path.split(input_path)
    name, _ = os.path.splitext(filename)

    output_path = os.path.join(folder, f"{name}_output_global.mp4")
    deoldify_full_path = os.path.join(folder, f"{name}_deoldify.mp4")
    results_frame_output_path = os.path.join(folder, f"{name}_yolo.mp4")

    # Cache check
    if os.path.exists(output_path) and not overwrite:
        print(f"[CACHE] Annotated video found: {output_path}")
        return output_path, deoldify_full_path, results_frame_output_path

    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"❌ Cannot open input video: {input_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Writers
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    deoldify_writer = cv2.VideoWriter(deoldify_full_path, fourcc, fps, (width, height))
    results_frame_writer = cv2.VideoWriter(results_frame_output_path, fourcc, fps, (width, height))

    try:
        with tqdm(total=total_frames, desc="Enhancing (global)", unit="frame") as pbar:
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                # Step 1: DeOldify full frame once
                frame_deoldify = deoldify_inference(frame_rgb)
                deoldify_writer.write(cv2.cvtColor(frame_deoldify, cv2.COLOR_RGB2BGR))

                # Step 2: YOLO prediction
                results = yolo_model.predict(frame_bgr, conf=CONF_THRESHOLD, verbose=False)
                results_frame = results[0].plot()
                results_frame_writer.write(results_frame)

                # Step 3: Apply YOLO masks (ignore boxes, use masks only)
                annotated = frame_rgb.copy()
                if results and results[0].masks is not None:
                    masks_raw = results[0].masks.data  # (N, h, w)

                    # Resize masks to match original frame size
                    masks_resized = torch.nn.functional.interpolate(
                        masks_raw.unsqueeze(1).float(),
                        size=(height, width),
                        mode="nearest"
                    ).squeeze(1).cpu().numpy()

                    classes = results[0].boxes.cls.cpu().numpy()

                    for mask, cls in zip(masks_resized, classes):
                        if int(cls) != 0:
                            continue  # only "person"

                        # Boolean mask aligned to full frame
                        mask_bool = mask > 0.5

                        # Blend from global DeOldify frame
                        annotated[mask_bool] = frame_deoldify[mask_bool]

                # Save final frame
                out_writer.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                pbar.update(1)

    except (Exception, KeyboardInterrupt) as e:
        print(f"[ERROR] DeOldify YOLO global pipeline interrupted: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        raise
    finally:
        cap.release()
        out_writer.release()
        deoldify_writer.release()
        results_frame_writer.release()
        import gc
        torch.cuda.empty_cache()
        gc.collect()

    print(f"[INFO] Annotated global video saved: {output_path}")
    print(f"[INFO] Full-frame DeOldify video saved: {deoldify_full_path}")
    print(f"[INFO] YOLO debug video saved: {results_frame_output_path}")
    return output_path, deoldify_full_path, results_frame_output_path



def run_deoldify_yolo_global(input_path, overwrite=False):
    """
    Run DeOldify once per frame for the entire image.
    Then overlay YOLO masks (yellow, same coverage as YOLO .plot()).
    Saves:
        <name>_output_global.mp4  (final annotated with YOLO + DeOldify)
        <name>_deoldify.mp4       (DeOldify-only frames)
        <name>_yolo.mp4           (YOLO debug frames with yellow person masks)
    """
    folder, filename = os.path.split(input_path)
    name, _ = os.path.splitext(filename)

    output_path = os.path.join(folder, f"{name}_output_global.mp4")
    deoldify_full_path = os.path.join(folder, f"{name}_deoldify.mp4")
    results_frame_output_path = os.path.join(folder, f"{name}_yolo.mp4")

    # Cache check
    if os.path.exists(output_path) and not overwrite:
        print(f"[CACHE] Annotated video found: {output_path}")
        return output_path, deoldify_full_path, results_frame_output_path

    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"❌ Cannot open input video: {input_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Writers
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    deoldify_writer = cv2.VideoWriter(deoldify_full_path, fourcc, fps, (width, height))
    results_frame_writer = cv2.VideoWriter(results_frame_output_path, fourcc, fps, (width, height))

    try:
        with tqdm(total=total_frames, desc="Enhancing (global)", unit="frame") as pbar:
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                # Step 1: DeOldify full frame once
                frame_deoldify = deoldify_inference(frame_rgb)
                deoldify_writer.write(cv2.cvtColor(frame_deoldify, cv2.COLOR_RGB2BGR))

                # Step 2: YOLO prediction
                results = yolo_model.predict(frame_bgr, conf=CONF_THRESHOLD, verbose=False)

                # Start with YOLO's normal plotted frame (boxes, labels, etc.)
                results_frame = results[0].plot()

                # Overlay yellow masks with alpha blending (like YOLO does)
                if results and results[0].masks is not None:
                    masks_float = results[0].masks.data  # (N, h, w), values in [0,1]

                    # Resize to match frame (bilinear for smooth edges)
                    masks_resized = torch.nn.functional.interpolate(
                        masks_float.unsqueeze(1),
                        size=(height, width),
                        mode="bilinear",
                        align_corners=False
                    ).squeeze(1).cpu().numpy()

                    classes = results[0].boxes.cls.cpu().numpy()

                    for mask, cls in zip(masks_resized, classes):
                        if int(cls) != 0:
                            continue  # only "person"

                        # Use mask values directly as alpha
                        alpha_mask = np.expand_dims(mask, axis=-1)  # (H,W,1)
                        yellow = np.array([0, 255, 255], dtype=np.float32)

                        # Blend yellow with the results_frame wherever mask > 0
                        results_frame = (
                            alpha_mask * yellow +
                            (1 - alpha_mask) * results_frame
                        ).astype(np.uint8)

                results_frame_writer.write(results_frame)

                # Step 3: Apply YOLO masks to annotated frame
                annotated = frame_rgb.copy()
                if results and results[0].masks is not None:
                    for mask, cls in zip(masks_resized, classes):
                        if int(cls) != 0:
                            continue
                        # Same alpha blending for DeOldify
                        alpha_mask = np.expand_dims(mask, axis=-1)
                        annotated = (
                            alpha_mask * frame_deoldify +
                            (1 - alpha_mask) * annotated
                        ).astype(np.uint8)

                # Save final frame
                out_writer.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                pbar.update(1)

    except (Exception, KeyboardInterrupt) as e:
        print(f"[ERROR] DeOldify YOLO global pipeline interrupted: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        raise
    finally:
        cap.release()
        out_writer.release()
        deoldify_writer.release()
        results_frame_writer.release()
        import gc
        torch.cuda.empty_cache()
        gc.collect()

    print(f"[INFO] Annotated global video saved: {output_path}")
    print(f"[INFO] Full-frame DeOldify video saved: {deoldify_full_path}")
    print(f"[INFO] YOLO debug video saved: {results_frame_output_path}")
    return output_path, deoldify_full_path, results_frame_output_path





def run_deoldify_yolo_cached(input_path):
    """
    Wrapper with cache check.
    Returns only the main annotated output path (_output.mp4).
    """
    folder, filename = os.path.split(input_path)
    name, _ = os.path.splitext(filename)
    output_path = os.path.join(folder, f"{name}_output.mp4")

    if os.path.exists(output_path):
        print(f"[CACHE] Final video already exists: {output_path}")
        return output_path

    annotated, _, _ = run_deoldify_yolo_global(input_path)
    return annotated
