import os, cv2, json, uuid, time, pickle, requests, torch
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
from PIL import Image

# ==== CONFIG ====
YOLO_MODEL_PATH = "models/yolo11x-seg.pt"
CONF_THRESHOLD = 0.6
OUTPUT_ROOT = "outputs"
COMFY = "http://192.168.27.13:23476"
WORKFLOW_JSON = "ClothesArmsFaceNeck.json"
ENLARGE_PERCENT = 0.2
# =================


# ==== Color Transfer Config ====
USE_REGION_AB_AVERAGE = False   # if True, average DeOldify's A/B per region
USE_CUSTOM_AB = True           # if True, override with fixed values
CUSTOM_A = 128                  # fixed A (0–255, 128 = neutral)
CUSTOM_B = 115                  # fixed B (0–255, 128 = neutral)


def apply_region_color_transfer(f_in, f_de, mask):
    """
    Fusion helper: decides how to transfer colors from DeOldify to original frame.
    """
    if not (USE_REGION_AB_AVERAGE or USE_CUSTOM_AB):
        # 🔹 default pixelwise transfer
        out = f_in.copy()
        out[mask > 127] = f_de[mask > 127]
        return out

    # Convert to LAB
    lab_in = cv2.cvtColor(f_in, cv2.COLOR_BGR2LAB)
    lab_de = cv2.cvtColor(f_de, cv2.COLOR_BGR2LAB)
    out_lab = lab_in.copy()

    # Find connected regions
    num_labels, labels = cv2.connectedComponents((mask > 127).astype(np.uint8))
    for lbl in range(1, num_labels):
        region_mask = (labels == lbl)

        if USE_CUSTOM_AB:
            meanA, meanB = CUSTOM_A, CUSTOM_B
        else:
            A_vals = lab_de[...,1][region_mask]
            B_vals = lab_de[...,2][region_mask]
            if A_vals.size == 0: continue
            meanA, meanB = int(np.mean(A_vals)), int(np.mean(B_vals))

        out_lab[...,1][region_mask] = meanA
        out_lab[...,2][region_mask] = meanB

    fused_bgr = cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)
    out = f_in.copy()
    out[mask > 127] = fused_bgr[mask > 127]
    return out


output1 = "clothes"
output2 = "arms"
output3 = "neck"
output4 = "faces"

# output1 = "face"
# output2 = "bike"
# output3 = "car"
# output4 = "scooty"

#######################

# Prompts + thresholds (can be edited dynamically or even loaded from a file/CLI args)
SEGMENT_CONFIG = {
    "2": {"prompt": output1, "threshold": 0.30},
    "6": {"prompt": output2,    "threshold": 0.30},
    "8": {"prompt": output3,   "threshold": 0.30},
    "10": {"prompt": output4,  "threshold": 0.30}
}





# ---- Setup models ----
device.set(device=DeviceId.GPU0)
colorizer = get_image_colorizer(artistic=True)
device_str = "cuda" if torch.cuda.is_available() else "cpu"
yolo_model = YOLO(YOLO_MODEL_PATH).to(device_str)


# ---- DeOldify ----
def deoldify_inference(frame_rgb):
    pil_img = Image.fromarray(frame_rgb).convert("RGB")
    ret = colorizer.get_transformed_image(pil_img, render_factor=16, post_process=True)
    return np.array(ret)


# ---- ComfyUI Helpers ----
def upload_image_to_comfy(local_path, server=COMFY, *, dest_name=None, folder_type="input"):
    if dest_name is None:
        dest_name = os.path.basename(local_path)
    with open(local_path, "rb") as f:
        files = {"image": (dest_name, f, "image/png")}
        data = {"type": folder_type, "overwrite": "true"}
        r = requests.post(f"{server}/upload/image", files=files, data=data, timeout=60)
        r.raise_for_status()
    return dest_name

def queue_prompt(prompt_dict, server=COMFY):
    client_id = str(uuid.uuid4())
    r = requests.post(f"{server}/prompt", json={"prompt": prompt_dict, "client_id": client_id}, timeout=120)
    r.raise_for_status()
    return r.json().get("prompt_id", client_id)

def get_history(prompt_id, server=COMFY):
    r = requests.get(f"{server}/history/{prompt_id}", timeout=60)
    r.raise_for_status()
    return r.json()

def download_image(filename, server=COMFY, folder_type="output", subfolder="", to_path=None, save_dir=None):
    if save_dir is None:
        save_dir = os.path.join(OUTPUT_ROOT, "comfy_downloads")
    os.makedirs(save_dir, exist_ok=True)
    params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    r = requests.get(f"{server}/view", params=params, timeout=60)
    r.raise_for_status()
    if to_path is None:
        to_path = os.path.join(save_dir, filename)
    with open(to_path, "wb") as f:
        f.write(r.content)
    return to_path

def run_multi_masks(frame_path, comfy_server=COMFY):
    uploaded = upload_image_to_comfy(frame_path, server=comfy_server)
    with open(WORKFLOW_JSON, "r") as f:
        prompt = json.load(f)

    # patch LoadImage
    for node in prompt.values():
        if node.get("class_type", "").lower() == "loadimage":
            node["inputs"]["image"] = uploaded

    # Patch prompts + thresholds dynamically
    for node_id, cfg in SEGMENT_CONFIG.items():
        if node_id in prompt:
            prompt[node_id]["inputs"]["prompt"] = cfg["prompt"]
            prompt[node_id]["inputs"]["threshold"] = cfg["threshold"]

    prompt_id = queue_prompt(prompt, server=comfy_server)
    deadline = time.time() + 600
    masks = {}

    # mapping node_id -> semantic name
    id_to_name = {
        "12": output1,
        "13": output2,
        "14": output3,
        "15": output4
    }

    while time.time() < deadline:
        hist = get_history(prompt_id, server=comfy_server)
        item = hist.get(prompt_id)
        if item and "outputs" in item:
            for node_id, node_out in item["outputs"].items():
                for im in node_out.get("images", []):
                    fn = im["filename"]
                    sub = im.get("subfolder", "")
                    typ = im.get("type", "output")
                    out_path = os.path.join(os.path.dirname(frame_path),
                                            f"{id_to_name.get(node_id, node_id)}_{os.path.basename(frame_path)}")
                    dl_path = download_image(fn, server=comfy_server,
                                             subfolder=sub, folder_type=typ, to_path=out_path)
                    masks[id_to_name.get(node_id, node_id)] = dl_path
        if masks:
            break
        time.sleep(0.5)

    return masks


# ---- Utility ----
def ensure_binary(img, size):
    if img is None:
        return np.zeros(size, dtype=np.uint8)
    img = cv2.resize(img, size)
    _, bin_img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    return bin_img


# ---- Stage 1: YOLO ----
def run_yolo(input_path, out_dir, name):
    yolo_path = os.path.join(out_dir, f"{name}_yolo_results.pkl")
    if os.path.exists(yolo_path):
        with open(yolo_path, "rb") as f:
            return pickle.load(f)

    results_per_frame = []
    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    with tqdm(total=total_frames, desc="YOLO", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret: break
            results = yolo_model.predict(frame, conf=CONF_THRESHOLD, verbose=False, device=device_str)
            results_per_frame.append({
                "boxes": results[0].boxes.xyxy.cpu().numpy(),
                "conf": results[0].boxes.conf.cpu().numpy(),
                "cls": results[0].boxes.cls.cpu().numpy()
            })
            pbar.update(1)
    cap.release()

    with open(yolo_path, "wb") as f:
        pickle.dump(results_per_frame, f)
    return results_per_frame




def run_yolo(input_path, out_dir, name):
    """
    Run YOLO on a video, save both detection results (.pkl)
    and an annotated output video (.mp4).
    """
    # ---- Output paths ----
    yolo_path = os.path.join(out_dir, f"{name}_yolo_results.pkl")
    yolo_video_path = os.path.join(out_dir, f"{name}_yolo_output.mp4")

    # ---- Use cache if available ----
    if os.path.exists(yolo_path) and os.path.exists(yolo_video_path):
        with open(yolo_path, "rb") as f:
            return pickle.load(f)

    # ---- Setup video I/O ----
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(out_dir, exist_ok=True)
    writer = cv2.VideoWriter(yolo_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # ---- Detection loop ----
    results_per_frame = []
    with tqdm(total=total_frames, desc="YOLO", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO inference
            results = yolo_model.predict(frame, conf=CONF_THRESHOLD, verbose=False, device=device_str)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()

            # Store frame-level results
            results_per_frame.append({
                "boxes": boxes,
                "conf": confs,
                "cls": classes
            })

            # Draw detections on frame
            annotated_frame = results[0].plot()  # YOLO's built-in plot()
            writer.write(annotated_frame)

            pbar.update(1)

    # ---- Cleanup ----
    cap.release()
    writer.release()

    # ---- Save detection results ----
    with open(yolo_path, "wb") as f:
        pickle.dump(results_per_frame, f)

    print(f"[YOLO] Saved annotated video → {yolo_video_path}")
    print(f"[YOLO] Saved detection results → {yolo_path}")
    return results_per_frame


# ---- Stage 2: DeOldify ----
def run_deoldify(input_path, out_dir, name):
    deoldify_path = os.path.join(out_dir, f"{name}_deoldify.mp4")
    if os.path.exists(deoldify_path):
        return deoldify_path

    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(deoldify_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="DeOldify", unit="frame") as pbar:
        while True:
            ret, f_bgr = cap.read()
            if not ret: break
           # f_rgb = cv2.cvtColor(f_bgr, cv2.COLOR_BGR2RGB)
            f_gray = cv2.cvtColor(f_bgr, cv2.COLOR_BGR2GRAY)
            f_rgb = cv2.cvtColor(f_gray, cv2.COLOR_GRAY2RGB)
            deold = deoldify_inference(f_rgb)
            writer.write(cv2.cvtColor(deold, cv2.COLOR_RGB2BGR))
            pbar.update(1)
    cap.release()
    writer.release()
    return deoldify_path



import os
import subprocess

def run_deoldify(input_path, out_dir, name, first_path=None):
    """
    Run U-Net video colorization via subprocess (cached).
    
    Parameters:
        input_path (str): Path to grayscale input video
        out_dir (str): Directory to save output video
        name (str): Base name for output video
        first_path (str, optional): Placeholder for compatibility with old API
    
    Returns:
        str: Path to the colorized video
    """
    unet_video_path = os.path.join(out_dir, f"{name}_deoldify.mp4")
    if os.path.exists(unet_video_path):
        return unet_video_path

    # 🔹 Hardcoded generator weights path
    generator_weights = "models/best_weights_epoch_0004.weights.h5"

    # 🔹 Call the existing cached subprocess wrapper
    from Utils.main_utils import run_unet_colorization_cached_subprocess
    unet_video_path = run_unet_colorization_cached_subprocess(
        input_bw_video=input_path,
        unet_weights=generator_weights,
        first_path=first_path or input_path
    )

    return unet_video_path


# ---- Stage 3: Precompute SAM masks (MaskA + MaskB videos) ----
def run_sam_masks(input_path, out_dir, name):
    maskA_path = os.path.join(out_dir, f"{name}_maskA.mp4")
    maskB_path = os.path.join(out_dir, f"{name}_maskB.mp4")
    if os.path.exists(maskA_path) and os.path.exists(maskB_path):
        return maskA_path, maskB_path

    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writerA = cv2.VideoWriter(maskA_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    writerB = cv2.VideoWriter(maskB_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    with tqdm(total=total_frames, desc="SAM", unit="frame") as pbar:
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_file = os.path.join(out_dir, f"frame_{idx:06d}.png")
            cv2.imwrite(frame_file, frame)

            masks = run_multi_masks(frame_file, comfy_server=COMFY)
            mask_imgs = {}
            for k, v in masks.items():
                img = cv2.imread(v, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    mask_imgs[k] = ensure_binary(img, (width, height))

            mask_out1 = mask_imgs.get(output1, np.zeros((height, width), np.uint8))
            mask_out2 = mask_imgs.get(output2, np.zeros((height, width), np.uint8))
            mask_out3 = mask_imgs.get(output3, np.zeros((height, width), np.uint8))
            mask_out4 = mask_imgs.get(output4, np.zeros((height, width), np.uint8))
            
            others = cv2.bitwise_or(mask_out2, cv2.bitwise_or(mask_out3, mask_out4))
            maskA = cv2.bitwise_and(mask_out1, cv2.bitwise_not(others))  # exclusive output1
            maskB = cv2.bitwise_or(mask_out2, mask_out3)                # output2 + output3


            writerA.write(cv2.cvtColor(maskA, cv2.COLOR_GRAY2BGR))
            writerB.write(cv2.cvtColor(maskB, cv2.COLOR_GRAY2BGR))

            idx += 1
            pbar.update(1)

    cap.release()
    writerA.release()
    writerB.release()
    return maskA_path, maskB_path


def run_sam_masks(input_path, out_dir, name, yolo_results):
    maskA_path = os.path.join(out_dir, f"{name}_maskA.mp4")
    maskB_path = os.path.join(out_dir, f"{name}_maskB.mp4")
    if os.path.exists(maskA_path) and os.path.exists(maskB_path):
        return maskA_path, maskB_path

    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writerA = cv2.VideoWriter(maskA_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    writerB = cv2.VideoWriter(maskB_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    with tqdm(total=total_frames, desc="SAM", unit="frame") as pbar:
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Skip if YOLO found nothing
            if idx >= len(yolo_results) or len(yolo_results[idx]["boxes"]) == 0:
                writerA.write(np.zeros((height, width, 3), dtype=np.uint8))
                writerB.write(np.zeros((height, width, 3), dtype=np.uint8))
                idx += 1
                pbar.update(1)
                continue

            frame_file = os.path.join(out_dir, f"frame_{idx:06d}.png")
            cv2.imwrite(frame_file, frame)
            masks = run_multi_masks(frame_file, comfy_server=COMFY)

            mask_imgs = {}
            for k, v in masks.items():
                img = cv2.imread(v, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    mask_imgs[k] = ensure_binary(img, (width, height))

            mask_out1 = mask_imgs.get(output1, np.zeros((height, width), np.uint8))
            mask_out2 = mask_imgs.get(output2, np.zeros((height, width), np.uint8))
            mask_out3 = mask_imgs.get(output3, np.zeros((height, width), np.uint8))
            mask_out4 = mask_imgs.get(output4, np.zeros((height, width), np.uint8))

            others = cv2.bitwise_or(mask_out2, cv2.bitwise_or(mask_out3, mask_out4))
            maskA = cv2.bitwise_and(mask_out1, cv2.bitwise_not(others))
            others_B = cv2.bitwise_or(mask_out1, mask_out4)
            maskB = cv2.bitwise_and(cv2.bitwise_or(mask_out2, mask_out3), cv2.bitwise_not(others_B))

            # Apply YOLO bounding box restriction
            yolo_mask = np.zeros((height, width), np.uint8)
            for box in yolo_results[idx]["boxes"]:
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                yolo_mask[y1:y2, x1:x2] = 255

            maskA = cv2.bitwise_and(maskA, yolo_mask)
            maskB = cv2.bitwise_and(maskB, yolo_mask)

            # If both masks are empty → discard
            if np.count_nonzero(maskA) == 0 and np.count_nonzero(maskB) == 0:
                writerA.write(np.zeros((height, width, 3), dtype=np.uint8))
                writerB.write(np.zeros((height, width, 3), dtype=np.uint8))
            else:
                writerA.write(cv2.cvtColor(maskA, cv2.COLOR_GRAY2BGR))
                writerB.write(cv2.cvtColor(maskB, cv2.COLOR_GRAY2BGR))

            idx += 1
            pbar.update(1)

    cap.release()
    writerA.release()
    writerB.release()
    return maskA_path, maskB_path


# ---- Stage 4: Fusion ----
def run_fusion(input_path, out_dir, name, yolo_results, deoldify_path, maskA_path, maskB_path):
    finalA_path = os.path.join(out_dir, f"{name}_finalA.mp4")
    finalB_path = os.path.join(out_dir, f"{name}_finalB.mp4")
    if os.path.exists(finalA_path) and os.path.exists(finalB_path):
        return finalA_path, finalB_path

    cap_in = cv2.VideoCapture(input_path)
    cap_de = cv2.VideoCapture(deoldify_path)
    cap_A  = cv2.VideoCapture(maskA_path)
    cap_B  = cv2.VideoCapture(maskB_path)

    fps = int(cap_in.get(cv2.CAP_PROP_FPS)) or 25
    width = int(cap_in.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writerA = cv2.VideoWriter(finalA_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    writerB = cv2.VideoWriter(finalB_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_idx = 0
    with tqdm(total=len(yolo_results), desc="Fusion", unit="frame") as pbar:
        while frame_idx < len(yolo_results):
            ret_in, f_in = cap_in.read()
            ret_de, f_de = cap_de.read()
            ret_A, f_A = cap_A.read()
            ret_B, f_B = cap_B.read()
            if not (ret_in and ret_de and ret_A and ret_B): break

            grayA = cv2.cvtColor(f_A, cv2.COLOR_BGR2GRAY)
            grayB = cv2.cvtColor(f_B, cv2.COLOR_BGR2GRAY)

            _, maskA = cv2.threshold(grayA, 1, 255, cv2.THRESH_BINARY)
            _, maskB = cv2.threshold(grayB, 1, 255, cv2.THRESH_BINARY)

            # YOLO restriction
            yolo_mask = np.zeros((height, width), dtype=np.uint8)
            for box, conf, cls in zip(yolo_results[frame_idx]["boxes"],
                                      yolo_results[frame_idx]["conf"],
                                      yolo_results[frame_idx]["cls"]):
                if int(cls) != 0 or conf < CONF_THRESHOLD: continue
                x1, y1, x2, y2 = map(int, box)
                dx = int((x2 - x1) * ENLARGE_PERCENT)
                dy = int((y2 - y1) * ENLARGE_PERCENT)
                x1 = max(0, x1 - dx); y1 = max(0, y1 - dy)
                x2 = min(width, x2 + dx); y2 = min(height, y2 + dy)
                yolo_mask[y1:y2, x1:x2] = 255

            maskA = cv2.bitwise_and(maskA, yolo_mask)
            maskB = cv2.bitwise_and(maskB, yolo_mask)

            # fA = f_in.copy(); fA[maskA > 127] = f_de[maskA > 127]
            # fB = f_in.copy(); fB[maskB > 127] = f_de[maskB > 127]

            fA = apply_region_color_transfer(f_in, f_de, maskA)
            fB = apply_region_color_transfer(f_in, f_de, maskB)


            writerA.write(fA)
            writerB.write(fB)

            frame_idx += 1
            pbar.update(1)

    cap_in.release(); cap_de.release(); cap_A.release(); cap_B.release()
    writerA.release(); writerB.release()
    return finalA_path, finalB_path


# ---- Main Pipeline ----
def process_video_with_multi_masks(input_path):
    folder, fname = os.path.split(input_path)
    name, _ = os.path.splitext(fname)
    OUTPUT_ROOT = folder
    out_dir = os.path.join(OUTPUT_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)

    yolo_results = run_yolo(input_path, out_dir, name)
    deoldify_path = run_deoldify(input_path, out_dir, name)
    maskA_path, maskB_path = run_sam_masks(input_path, out_dir, name, yolo_results)
    finalA_path, finalB_path = run_fusion(input_path, out_dir, name, yolo_results,
                                          deoldify_path, maskA_path, maskB_path)

    print(f"[INFO] Outputs written to {out_dir}")
    return {
        "deoldify": deoldify_path,
        "maskA": maskA_path,
        "maskB": maskB_path,
        "finalA": finalA_path,
        "finalB": finalB_path
    }


# ---- Example ----
# if __name__ == "__main__":
#     input_video = "input_videos/thatha_manavadu_test.mp4"
#     outputs = process_video_with_multi_masks(input_video)
#     print("Pipeline outputs:")
#     for k, v in outputs.items():
#         print(f" - {k}: {v}")


def process_video_cached(input_path):
    """
    Cached wrapper around process_video().
    Returns only the final fusion video path.
    """
    folder, fname = os.path.split(input_path)
    name, _ = os.path.splitext(fname)

    # Same folder as input
    OUTPUT_ROOT = folder  

    out_dir = os.path.join(OUTPUT_ROOT, name)
    final_path = os.path.join(out_dir, f"{name}_final.mp4")

    if os.path.exists(final_path):
        print(f"[CACHE] Final output exists: {final_path}")
        return final_path

    outputs = process_video_with_multi_masks(input_path)
    return outputs["finalA"]
