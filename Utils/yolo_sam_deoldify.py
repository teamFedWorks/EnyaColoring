import os
import cv2
import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Using:", torch.cuda.get_device_name(0))
import numpy as np
import pickle
from tqdm import tqdm
from ultralytics import YOLO
from deoldify.visualize import get_image_colorizer
from deoldify import device
from deoldify.device_id import DeviceId
from PIL import Image
import uuid, json, requests, time



# ==== CONFIG ====
YOLO_MODEL_PATH = "models/yolo11x-seg.pt"
CONF_THRESHOLD = 0.6
OUTPUT_ROOT = "outputs"
COMFY = "http://192.168.27.13:23476"
WORKFLOW_JSON = "ClothesDetect_api.json"

# ---- NEW CONFIG ----
BBOX_ENLARGE = 0.2       # enlarge bbox by 20%
TOP_K_BBOX = 3           # number of top boxes per frame to run SAM
GDINO_PROMPT = "clothes" # grounding dino prompt
GDINO_THRESHOLD = 0.30   # grounding dino threshold
# =================


def patch_groundingdino_node(prompt_dict, new_prompt=None, new_threshold=None):
    """Patch GroundingDinoSAMSegment node with new prompt/threshold values."""
    for node in prompt_dict.values():
        if node.get("class_type", "").lower().startswith("groundingdinosamsegment"):
            if new_prompt is not None:
                node["inputs"]["prompt"] = new_prompt
            if new_threshold is not None:
                node["inputs"]["threshold"] = new_threshold
            return True
    return False



# =================

# ---- Setup DeOldify ----
device.set(device=DeviceId.GPU0)
colorizer = get_image_colorizer(artistic=True)

# ---- Setup YOLO ----
device_str = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device_str}")
yolo_model = YOLO(YOLO_MODEL_PATH).to(device_str)


# ---- DeOldify inference ----
def deoldify_inference(frame_rgb):
    pil_img = Image.fromarray(frame_rgb).convert("RGB")
    ret = colorizer.get_transformed_image(pil_img, render_factor=16, post_process=True)
    return np.array(ret)


# ---- ComfyUI helpers ----
def upload_image_to_comfy(local_path, server=COMFY, *, dest_name=None, folder_type="input"):
    if dest_name is None:
        dest_name = os.path.basename(local_path)
    with open(local_path, "rb") as f:
        files = {"image": (dest_name, f, "image/png")}
        data = {"type": folder_type, "overwrite": "true"}
        r = requests.post(f"{server}/upload/image", files=files, data=data, timeout=60)
        r.raise_for_status()
    return dest_name





def patch_loadimage_node(prompt_dict, new_filename):
    for node in prompt_dict.values():
        if node.get("class_type","").lower() == "loadimage":
            node["inputs"]["image"] = new_filename
            return True
    return False



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


def run_sam_on_frame(frame_path, comfy_server=COMFY):
    """Send one frame (crop) through ComfyUI workflow and return saved mask path."""
    uploaded = upload_image_to_comfy(frame_path, server=comfy_server)

    with open(WORKFLOW_JSON, "r") as f:
        prompt = json.load(f)

    # Patch LoadImage node
    if not patch_loadimage_node(prompt, uploaded):
        raise RuntimeError("Could not patch LoadImage node in workflow JSON.")

    # 🔹 Patch GroundingDino node with dynamic prompt & threshold
    patch_groundingdino_node(prompt, new_prompt=GDINO_PROMPT, new_threshold=GDINO_THRESHOLD)

    prompt_id = queue_prompt(prompt, server=comfy_server)
    deadline = time.time() + 600
    seg_path = None

    while time.time() < deadline:
        hist = get_history(prompt_id, server=comfy_server)
        item = hist.get(prompt_id)
        if item and "outputs" in item:
            for node_out in item["outputs"].values():
                for im in node_out.get("images", []):
                    fn = im["filename"]
                    sub = im.get("subfolder", "")
                    typ = im.get("type", "output")
                    base = os.path.splitext(os.path.basename(frame_path))[0]
                    save_dir = os.path.dirname(frame_path)
                    out_path = os.path.join(save_dir, f"ComfyUI_{base}.png")
                    seg_path = download_image(fn, server=comfy_server,
                                              subfolder=sub, folder_type=typ,
                                              to_path=out_path)
                    break
        if seg_path:
            break
        time.sleep(0.5)

    if not seg_path:
        raise RuntimeError(f"No outputs from ComfyUI for {frame_path}")

    return seg_path

# ---- Stage 3: SAM with YOLO BBoxes ----
def run_sam(input_path, out_dir, name, yolo_results):
    sam_frames_dir = os.path.join(out_dir, f"{name}_sam_frames")
    os.makedirs(sam_frames_dir, exist_ok=True)
    sam_path = os.path.join(out_dir, f"{name}_sam.mp4")

    if os.path.exists(sam_path):
        print(f"[CACHE] Using cached SAM video: {sam_path}")
        return sam_path

    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_idx = 0
    with tqdm(total=total_frames, desc="SAM with YOLO", unit="frame") as pbar:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            results = yolo_results[frame_idx]
            boxes, confs, clses = results["boxes"], results["conf"], results["cls"]

            order = np.argsort(confs)[::-1][:TOP_K_BBOX]
            masks_for_frame = []

            for i in order:
                if int(clses[i]) != 0 or confs[i] < CONF_THRESHOLD:
                    continue
                x1, y1, x2, y2 = map(int, boxes[i])

                # enlarge box
                bw = x2 - x1
                bh = y2 - y1
                x1 = max(0, int(x1 - BBOX_ENLARGE * bw))
                y1 = max(0, int(y1 - BBOX_ENLARGE * bh))
                x2 = min(width, int(x2 + BBOX_ENLARGE * bw))
                y2 = min(height, int(y2 + BBOX_ENLARGE * bh))

                crop = frame_bgr[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                crop_path = os.path.join(sam_frames_dir, f"frame_{frame_idx:06d}_box{i}.png")
                cv2.imwrite(crop_path, crop)

                try:
                    seg_path = run_sam_on_frame(crop_path, comfy_server=COMFY)
                    seg_img = cv2.imread(seg_path)
                    seg_resized = cv2.resize(seg_img, (x2 - x1, y2 - y1))
                    mask = np.zeros((height, width), dtype=np.uint8)
                    mask[y1:y2, x1:x2] = cv2.cvtColor(seg_resized, cv2.COLOR_BGR2GRAY)
                    masks_for_frame.append(mask)
                except Exception as e:
                    print(f"⚠️ SAM failed on frame {frame_idx}, box {i}: {e}")
                finally:
                    # cleanup crop + box-level SAM output
                    if os.path.exists(crop_path):
                        os.remove(crop_path)
                    box_seg = crop_path.replace(".png", "").replace("frame_", "ComfyUI_frame_") + ".png"
                    if os.path.exists(box_seg):
                        os.remove(box_seg)

            # always save a mask (blank if no detections)
            if masks_for_frame:
                final_mask = np.zeros((height, width), dtype=np.uint8)
                for m in masks_for_frame:
                    final_mask = cv2.bitwise_or(final_mask, m)
            else:
                final_mask = np.zeros((height, width), dtype=np.uint8)

            out_path = os.path.join(sam_frames_dir, f"ComfyUI_frame_{frame_idx:06d}.png")
            cv2.imwrite(out_path, final_mask)

            frame_idx += 1
            pbar.update(1)
    cap.release()

    # build video ONLY from final per-frame masks
    sam_files = sorted([
        f for f in os.listdir(sam_frames_dir)
        if f.startswith("ComfyUI_frame_") and "_box" not in f
    ])
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(sam_path, fourcc, fps, (width, height))
    for seg_file in tqdm(sam_files, desc="Building SAM video", unit="frame"):
        img = cv2.imread(os.path.join(sam_frames_dir, seg_file))
        writer.write(cv2.resize(img, (width, height)))
    writer.release()
    return sam_path


# ---- Stage 1: YOLO ----
def run_yolo(input_path, out_dir, name):
    yolo_path = os.path.join(out_dir, f"{name}_yolo.mp4")
    results_path = os.path.join(out_dir, f"{name}_yolo_results.pkl")

    if os.path.exists(yolo_path) and os.path.exists(results_path):
        print(f"[CACHE] Using cached YOLO + results: {yolo_path}")
        with open(results_path, "rb") as f:
            results_per_frame = pickle.load(f)
        return yolo_path, results_per_frame

    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(yolo_path, fourcc, fps, (width, height))

    results_per_frame = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    with tqdm(total=total_frames, desc="YOLO", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = yolo_model.predict(frame, conf=CONF_THRESHOLD, verbose=False, device=device_str)
            writer.write(results[0].plot())
            results_per_frame.append({
                "boxes": results[0].boxes.xyxy.cpu().numpy(),
                "conf": results[0].boxes.conf.cpu().numpy(),
                "cls": results[0].boxes.cls.cpu().numpy()
            })
            pbar.update(1)

    cap.release()
    writer.release()

    with open(results_path, "wb") as f:
        pickle.dump(results_per_frame, f)

    return yolo_path, results_per_frame


# ---- Stage 2: DeOldify ----
def run_deoldify(input_path, out_dir, name):
    deoldify_path = os.path.join(out_dir, f"{name}_deoldify.mp4")
    if os.path.exists(deoldify_path):
        print(f"[CACHE] Using cached DeOldify: {deoldify_path}")
        return deoldify_path

    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(deoldify_path, fourcc, fps, (width, height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    with tqdm(total=total_frames, desc="DeOldify", unit="frame") as pbar:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            deold = deoldify_inference(frame_rgb)
            writer.write(cv2.cvtColor(deold, cv2.COLOR_RGB2BGR))
            pbar.update(1)
    cap.release()
    writer.release()
    return deoldify_path


# ---- Stage 4: Fusion ----
def run_fusion(input_path, out_dir, name, yolo_results, deoldify_path, sam_path):
    fusion_path = os.path.join(out_dir, f"{name}_final.mp4")
    if os.path.exists(fusion_path):
        print(f"[CACHE] Using cached Fusion: {fusion_path}")
        return fusion_path

    cap_input = cv2.VideoCapture(input_path)
    cap_deold = cv2.VideoCapture(deoldify_path)
    cap_sam = cv2.VideoCapture(sam_path)

    fps = int(cap_input.get(cv2.CAP_PROP_FPS)) or 25
    width = int(cap_input.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_input.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(fusion_path, fourcc, fps, (width, height))

    total_frames = int(min(
        len(yolo_results),
        cap_input.get(cv2.CAP_PROP_FRAME_COUNT),
        cap_deold.get(cv2.CAP_PROP_FRAME_COUNT),
        cap_sam.get(cv2.CAP_PROP_FRAME_COUNT)
    ))

    frame_idx = 0
    with tqdm(total=total_frames, desc="Fusion", unit="frame") as pbar:
        while frame_idx < total_frames:
            ret_in, frame_in = cap_input.read()
            ret_deold, frame_deold = cap_deold.read()
            ret_sam, frame_sam = cap_sam.read()
            if not (ret_in and ret_deold and ret_sam):
                break

            gray = cv2.cvtColor(frame_sam, cv2.COLOR_BGR2GRAY)
            _, sam_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            mask_bool = sam_mask > 127

            fusion_frame = frame_in.copy()
            fusion_frame[mask_bool] = frame_deold[mask_bool]

            writer.write(fusion_frame)
            frame_idx += 1
            pbar.update(1)

    cap_input.release()
    cap_deold.release()
    cap_sam.release()
    writer.release()

    print(f"[INFO] Fusion video saved: {fusion_path}")
    return fusion_path


# ---- Main Pipeline ----
# def process_video(input_path):
#     folder, fname = os.path.split(input_path)
#     name, _ = os.path.splitext(fname)
#     out_dir = os.path.join(OUTPUT_ROOT, name)
#     os.makedirs(out_dir, exist_ok=True)

#     yolo_path, yolo_results = run_yolo(input_path, out_dir, name)
#     deoldify_path = run_deoldify(input_path, out_dir, name)
#     sam_path = run_sam(input_path, out_dir, name, yolo_results)
#     fusion_path = run_fusion(input_path, out_dir, name, yolo_results, deoldify_path, sam_path)

#     print(f"[INFO] Outputs written to {out_dir}")
#     return {
#         "yolo": yolo_path,
#         "deoldify": deoldify_path,
#         "sam": sam_path,
#         "final": fusion_path
#     }


def process_video(input_path, OUTPUT_ROOT):
    folder, fname = os.path.split(input_path)
    name, _ = os.path.splitext(fname)
    out_dir = os.path.join(OUTPUT_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)

    yolo_path, yolo_results = run_yolo(input_path, out_dir, name)
    deoldify_path = run_deoldify(input_path, out_dir, name)
    sam_path = run_sam(input_path, out_dir, name, yolo_results)
    fusion_path = run_fusion(input_path, out_dir, name, yolo_results, deoldify_path, sam_path)

    print(f"[INFO] Outputs written to {out_dir}")
    return {
        "yolo": yolo_path,
        "deoldify": deoldify_path,
        "sam": sam_path,
        "final": fusion_path
    }



# def process_video_cached(input_path):
#     """
#     Cached wrapper around process_video().
#     Returns only the final fusion video path.
#     """
#     folder, fname = os.path.split(input_path)
#     name, _ = os.path.splitext(fname)
#     out_dir = os.path.join(OUTPUT_ROOT, name)
#     final_path = os.path.join(out_dir, f"{name}_final.mp4")
#     if os.path.exists(final_path):
#         print(f"[CACHE] Final output exists: {final_path}")
#         return final_path

#     outputs = process_video(input_path)
#     return outputs["final"]


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

    outputs = process_video(input_path, OUTPUT_ROOT)
    return outputs["final"]


# ---- Example ----
# if __name__ == "__main__":
#     input_video = "input_videos/thatha_manavadu_test.mp4"
#     outputs = process_video(input_video)
#     print("Pipeline outputs:")
#     for k, v in outputs.items():
#         print(f" - {k}: {v}")



