# --- Config ---
#COMFY = "http://192.168.27.12:8288"   # your ComfyUI server (headless or GUI)
COMFY = "http://localhost:8188"
WORKFLOW_JSON = "workflow_12.json"  # your uploaded graph
OUTPUT_DIR = "results"            # local folder to save downloads

# --- Imports ---
import os, io, json, time, uuid, pathlib, requests
from PIL import Image
from tqdm import tqdm

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- helper: compute target size and (optionally) resize before upload ---
from PIL import Image
import os, io, pathlib

MAX_SIDE = 512  # cap while preserving aspect ratio

def prepare_image_for_upload(local_path, *, max_side=MAX_SIDE, out_dir=OUTPUT_DIR):
    """
    Opens local_path, resizes if needed so that max(width, height) <= max_side (AR preserved),
    saves a PNG next to OUTPUT_DIR (to keep things tidy), and returns:
      (path_to_png, target_width, target_height)
    """
    img = Image.open(local_path).convert("RGB")
    w, h = img.size
    scale = 1.0
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
    target_w = max(1, int(round(w * scale)))
    target_h = max(1, int(round(h * scale)))
    if scale != 1.0:
        img = img.resize((target_w, target_h), Image.LANCZOS)

    # Save a temp PNG we’ll upload to ComfyUI
    base = pathlib.Path(local_path).stem
    resized_path = os.path.join(out_dir, f"{base}_rsz_{target_w}x{target_h}.png")
    img.save(resized_path, format="PNG")
    return resized_path, target_w, target_h

def prepare_image_for_upload(local_path, *, out_dir=OUTPUT_DIR):
    """
    Opens local_path, keeps full resolution (no resize),
    saves a PNG next to OUTPUT_DIR (to keep things tidy), and returns:
      (path_to_png, width, height)
    """
    img = Image.open(local_path).convert("RGB")
    w, h = img.size

    base = pathlib.Path(local_path).stem
    out_path = os.path.join(out_dir, f"{base}_orig_{w}x{h}.png")
    img.save(out_path, format="PNG")
    return out_path, w, h



# --- API Helpers ---

def upload_image_to_comfy(local_path, server=COMFY, *, dest_name=None, folder_type="input", subfolder="", overwrite=True):
    """
    POST /upload/image
    - folder_type: 'input' places image where LoadImage expects it.
    Returns the filename you should reference in the LoadImage node.
    """
    if dest_name is None:
        dest_name = pathlib.Path(local_path).name

    with open(local_path, "rb") as f:
        files = {"image": (dest_name, f, "image/png")}
        data = {"type": folder_type, "subfolder": subfolder, "overwrite": "true" if overwrite else "false"}
        r = requests.post(f"{server}/upload/image", files=files, data=data, timeout=120)
        r.raise_for_status()
    return dest_name  # ComfyUI references just the basename in LoadImage

def patch_prompt_text(prompt_dict, new_text):
    """
    Updates the first text-encoding node we find with the new prompt.
    Looks for keys like 'text' or 'prompt' in node inputs.
    """
    for node_id, node in prompt_dict.items():
        inputs = node.get("inputs", {})
        if "text" in inputs:
            inputs["text"] = new_text
            return True
        if "prompt" in inputs:
            inputs["prompt"] = new_text
            return True
    return False


def queue_prompt(prompt_dict, server=COMFY, client_id=None):
    """
    POST /prompt
    Returns prompt_id for polling /history/{prompt_id}.
    """
    if client_id is None:
        client_id = str(uuid.uuid4())
    r = requests.post(f"{server}/prompt", json={"prompt": prompt_dict, "client_id": client_id}, timeout=120)
    r.raise_for_status()
    return r.json().get("prompt_id", client_id)

def get_history(prompt_id, server=COMFY):
    """GET /history/{prompt_id} → JSON with outputs and image filenames."""
    r = requests.get(f"{server}/history/{prompt_id}", timeout=120)
    r.raise_for_status()
    return r.json()

def download_image(filename, server=COMFY, *, folder_type="output", subfolder="", to_path=None):
    """
    GET /view?filename=...&subfolder=...&type=output
    Returns local filepath saved under OUTPUT_DIR (default).
    """
    params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    r = requests.get(f"{server}/view", params=params, timeout=120)
    r.raise_for_status()
    if to_path is None:
        to_path = os.path.join(OUTPUT_DIR, filename)
    with open(to_path, "wb") as f:
        f.write(r.content)
    return to_path

def patch_loadimage_node(prompt_dict, new_filename):
    """
    Updates the first LoadImage-like node we find.
    ComfyUI 'LoadImage' nodes usually expose inputs.image (basename in /input).
    """
    for node_id, node in prompt_dict.items():
        ct = node.get("class_type", "")
        if ct.lower().replace(" ", "") in {"loadimage","imageloader"}:
            # Common schema: {"class_type": "LoadImage", "inputs": {"image": "some.png"}}
            node.setdefault("inputs", {})["image"] = new_filename
            return True
    return False

def patch_simple_knobs_(prompt_dict, *, seed=None, steps=None, cfg=None, width=None, height=None):
    """
    OPTIONAL: if you know your node ids, you can set them here.
    This tries to find likely KSampler/FluxGuidance/Scheduler nodes heuristically.
    Only patches when it finds a matching input field.
    """
    for node in prompt_dict.values():
        inputs = node.get("inputs", {})
        name = node.get("class_type", "").lower()
        if seed is not None and "seed" in inputs:
            inputs["seed"] = int(seed)
        if steps is not None and ("steps" in inputs or "num_inference_steps" in inputs):
            if "steps" in inputs:
                inputs["steps"] = int(steps)
            if "num_inference_steps" in inputs:
                inputs["num_inference_steps"] = int(steps)
        if cfg is not None:
            # Flux guidance scale is often 'guidance' or 'cfg' in custom nodes
            for k in ("guidance", "cfg", "guidance_scale"):
                if k in inputs:
                    inputs[k] = float(cfg)
        if width is not None and "width" in inputs:
            inputs["width"] = int(width)
        if height is not None and "height" in inputs:
            inputs["height"] = int(height)

def patch_simple_knobs(prompt_dict, *,
                       seed=None, steps=None,
                       cfg=None,              # KSampler CFG
                       flux_guidance=None,    # FluxGuidance guidance
                       width=None, height=None):
    """
    Patch common sampler/flux knobs separately.
    """
    for node in prompt_dict.values():
        inputs = node.get("inputs", {})
        ct = node.get("class_type", "").lower()

        # seed and steps
        if seed is not None and "seed" in inputs:
            inputs["seed"] = int(seed)
        if steps is not None and ("steps" in inputs or "num_inference_steps" in inputs):
            if "steps" in inputs:
                inputs["steps"] = int(steps)
            if "num_inference_steps" in inputs:
                inputs["num_inference_steps"] = int(steps)

        # KSampler cfg
        if cfg is not None and "ksampler" in ct and "cfg" in inputs:
            inputs["cfg"] = float(cfg)

        # FluxGuidance guidance
        if flux_guidance is not None and "fluxguidance" in ct and "guidance" in inputs:
            inputs["guidance"] = float(flux_guidance)

        # resolution
        if width is not None and "width" in inputs:
            inputs["width"] = int(width)
        if height is not None and "height" in inputs:
            inputs["height"] = int(height)


# --- Main: run workflow with a dynamic image ---

def run_workflow_with_image(image_path,
                            prompt_json_path=WORKFLOW_JSON,
                            *,
                            seed=2791063517, steps=10, cfg=1, flux_guidance=2.5, prompt_text=None, width=512, height=512,
                            comfy_server=COMFY):

    # 0) Prepare the image (resize if needed; get final W/H)
    prepped_path, target_w, target_h = prepare_image_for_upload(image_path)
    # 1) Upload the image so LoadImage can find it under 'input'
    uploaded_name = upload_image_to_comfy(prepped_path, server=comfy_server)

    # 2) Load your saved ComfyUI workflow JSON
    with open(prompt_json_path, "r") as f:
        prompt = json.load(f)

    # 3) Patch the LoadImage node to the new filename (and optional knobs)
    ok = patch_loadimage_node(prompt, uploaded_name)
    if not ok:
        raise RuntimeError("Could not find a LoadImage node to patch; check your graph JSON.")
    #patch_prompt_text(prompt, "Leave characters intact. Colorize this old photo with deep, vivid natural colors for clothes and background, use 1980s style fashion. Use soft natural skin tones and background colors. Inconsistent colors, bleeding, rusting, and blending is STRICTLY PROHIBITED.")
    patch_prompt_text(prompt, prompt_text)
    #patch_simple_knobs(prompt, seed=seed, steps=steps, cfg=cfg, width=target_w, height=target_h)
    patch_simple_knobs(
        prompt,
        seed=seed,
        steps=steps,
        cfg=cfg,                   # value for KSampler CFG
        flux_guidance=flux_guidance,         # set your FluxGuidance here
        width=target_w,
        height=target_h
    )

    # 4) Queue the prompt and poll history until it finishes
    prompt_id = queue_prompt(prompt, server=comfy_server)
    # Basic polling loop — you can also use websockets for live updates
    deadline = time.time() + 6000  # 10 min max
    output_files = []
    while time.time() < deadline:
        hist = get_history(prompt_id, server=comfy_server)
        # History payload nests outputs under the prompt_id
        item = hist.get(prompt_id)
        if item and "outputs" in item:
            # Collect all image outputs from any node that produced images
            for node_id, node_out in item["outputs"].items():
                images = node_out.get("images", [])
                for im in images:
                    fn = im.get("filename")
                    sub = im.get("subfolder", "")
                    typ = im.get("type", "output")
                    if fn:
                        local_path = download_image(fn, server=comfy_server, subfolder=sub, folder_type=typ)
                        if local_path not in output_files:
                            output_files.append(local_path)
            # If we have at least one image, consider it done
            if output_files:
                break
        time.sleep(0.5)

    if not output_files:
        raise RuntimeError("No images returned. Check the server logs and that your SaveImage path is valid.")

    return output_files

#seed=2791063517
def process_folder(input_folder, output_folder,
                   prompt_json_path=WORKFLOW_JSON,
                   *,
                   seed=2^24, steps=10,
                   cfg=1.0, flux_guidance=2.5,
                   comfy_server=COMFY,
                   prompt_text=None):
    print("processing folder using prompt_text, seed, steps, cfg, flux_guidance :", prompt_text, seed, steps, cfg, flux_guidance)
    os.makedirs(output_folder, exist_ok=True)

   # image_files = [f for f in os.listdir(input_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    image_files = sorted(
        [f for f in os.listdir(input_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    )

    # wrap with tqdm progress bar
    for fname in tqdm(image_files, desc="Processing frames", unit="frame"):
        in_path = os.path.join(input_folder, fname)

        # Only process image files
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        out_path = os.path.join(output_folder, fname)

        # Skip if already exists
        if os.path.exists(out_path):
            print(f"Skipping {fname}, already in output folder.")
            continue

        try:
            print(f"Processing {fname} ...")
            results = run_workflow_with_image(
                in_path,
                prompt_json_path=prompt_json_path,
                seed=seed,
                steps=steps,
                cfg=cfg,
                flux_guidance=flux_guidance,
                comfy_server=comfy_server,
                prompt_text=prompt_text
            )

            # Save the first result image under same name in output folder
            if results:
                # get original input size
                orig_w, orig_h = Image.open(in_path).size

                # open generated output
                out_img = Image.open(results[0])

                # resize output to match original input size
                out_img = out_img.resize((orig_w, orig_h), Image.LANCZOS)

                # save final image
                out_img.save(out_path)
                print(f"Saved → {out_path} ({orig_w}x{orig_h})")

        except Exception as e:
            print(f"❌ Error processing {fname}: {e}")

             




import os
import cv2
import math
import numpy as np
from tqdm import tqdm
from PIL import Image
from Utils.flux import run_workflow_with_image


def process_folder_concat_split(input_folder, output_folder,
                                *,
                                prompt_json_path=WORKFLOW_JSON,
                                seed=2 ** 24,
                                steps=10,
                                cfg=1.0,
                                flux_guidance=2.5,
                                comfy_server=COMFY,
                                prompt_text=None,
                                images_per_row=2,
                                total_images_per_combined=6):
    """
    Process frames in grouped batches (concatenation → colorization → splitting).
    Each combined image contains 'total_images_per_combined' frames in a grid.
    After colorization, each subframe is restored to original size and saved individually.
    """
    os.makedirs(output_folder, exist_ok=True)

    # --- Step 1: Gather input frames ---
    image_files = sorted(
        [f for f in os.listdir(input_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    )
    if not image_files:
        raise ValueError("❌ No input images found!")

    # --- Step 2: Get base dimensions ---
    sample_path = os.path.join(input_folder, image_files[0])
    sample_img = cv2.imread(sample_path)
    if sample_img is None:
        raise ValueError("❌ Failed to read first image.")
    base_h, base_w = sample_img.shape[:2]
    rows = math.ceil(total_images_per_combined / images_per_row)

    print(f"[INFO] Base frame: {base_w}x{base_h}, Grid: {rows}x{images_per_row}")

    # --- Step 3: Iterate in batches ---
    for i in tqdm(range(0, len(image_files), total_images_per_combined), desc="Batch Colorizing", unit="batch"):
        batch = image_files[i:i + total_images_per_combined]

        # Skip if all outputs already exist
        all_exist = all(
            os.path.exists(os.path.join(output_folder, f"frame_{i + idx:05d}.png"))
            for idx in range(len(batch))
        )
        if all_exist:
            print(f"[CACHE] Skipping batch {i // total_images_per_combined}, already processed.")
            continue

        # --- Step 4: Concatenate batch frames ---
        imgs = []
        for p in batch:
            img = cv2.imread(os.path.join(input_folder, p))
            if img is None:
                print(f"⚠️ Skipping unreadable image: {p}")
                continue
            imgs.append(img)

        if not imgs:
            continue

        new_w = base_w // images_per_row
        new_h = base_h // rows
        resized_imgs = [cv2.resize(im, (new_w, new_h)) for im in imgs]

        # pad black if fewer than required
        while len(resized_imgs) < total_images_per_combined:
            resized_imgs.append(np.zeros((new_h, new_w, 3), dtype=np.uint8))

        grid_rows = []
        for r in range(rows):
            start = r * images_per_row
            end = start + images_per_row
            row_imgs = resized_imgs[start:end]
            grid_rows.append(np.hstack(row_imgs))
        combined = np.vstack(grid_rows)

        combined_temp = os.path.join(output_folder, f"_combined_temp_{i:05d}.jpg")
        cv2.imwrite(combined_temp, combined)

        # --- Step 5: Run ComfyFlux workflow ---
        try:
            results = run_workflow_with_image(
                combined_temp,
                prompt_json_path=prompt_json_path,
                seed=seed,
                steps=steps,
                cfg=cfg,
                flux_guidance=flux_guidance,
                comfy_server=comfy_server,
                prompt_text=prompt_text
            )

            if not results:
                print(f"❌ No result for batch {i // total_images_per_combined}")
                continue

            colorized_path = results[0]
            colorized = cv2.imread(colorized_path)
            if colorized is None:
                print(f"❌ Failed to read output from ComfyFlux for batch {i}")
                continue

        except Exception as e:
            print(f"❌ Error in ComfyFlux for batch {i}: {e}")
            continue

        # --- Step 6: Split colorized grid back into frames ---
        h, w, _ = colorized.shape
        sub_w = w // images_per_row
        sub_h = h // rows

        frame_count = 0
        for r in range(rows):
            for c in range(images_per_row):
                if frame_count >= len(batch):
                    break
                y1, y2 = r * sub_h, (r + 1) * sub_h
                x1, x2 = c * sub_w, (c + 1) * sub_w
                sub_img = colorized[y1:y2, x1:x2]
                restored = cv2.resize(sub_img, (base_w, base_h))
                out_name = os.path.join(output_folder, f"frame_{i + frame_count:05d}.png")
                cv2.imwrite(out_name, restored)
                frame_count += 1

        print(f"✅ Batch {i // total_images_per_combined}: saved {frame_count} frames.")

        # --- Step 7: Cleanup temp file ---
        if os.path.exists(combined_temp):
            os.remove(combined_temp)

    print("🎉 All batches processed successfully!")





def process_folder_concat_split(input_folder, output_folder,
                                *,
                                prompt_json_path=WORKFLOW_JSON,
                                seed=2 ** 24,
                                steps=10,
                                cfg=1.0,
                                flux_guidance=2.5,
                                comfy_server=COMFY,
                                prompt_text=None,
                                images_per_row=2,
                                total_images_per_combined=6):
    """
    Process frames in grouped batches (concatenation → colorization → splitting).
    Each combined image contains 'total_images_per_combined' frames in a grid.
    After colorization, each subframe is restored to original size and saved individually.
    Aspect ratio of individual frames is preserved by centered padding.
    """

    import os, cv2, math, numpy as np
    from tqdm import tqdm
    from Utils.flux import run_workflow_with_image

    os.makedirs(output_folder, exist_ok=True)

    # --- Step 1: Gather input frames ---
    image_files = sorted(
        [f for f in os.listdir(input_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    )
    if not image_files:
        raise ValueError("❌ No input images found!")

    # --- Step 2: Get base dimensions ---
    sample_path = os.path.join(input_folder, image_files[0])
    sample_img = cv2.imread(sample_path)
    if sample_img is None:
        raise ValueError("❌ Failed to read first image.")
    base_h, base_w = sample_img.shape[:2]
    aspect_ratio = base_w / base_h
    rows = math.ceil(total_images_per_combined / images_per_row)

    print(f"[INFO] Base frame: {base_w}x{base_h}, Grid: {rows}x{images_per_row}")

    # --- Step 3: Iterate in batches ---
    for i in tqdm(range(0, len(image_files), total_images_per_combined),
                  desc="Batch Colorizing", unit="batch"):
        batch = image_files[i:i + total_images_per_combined]

        # Skip if all outputs already exist
        all_exist = all(
            os.path.exists(os.path.join(output_folder, f"frame_{i + idx:05d}.png"))
            for idx in range(len(batch))
        )
        if all_exist:
            print(f"[CACHE] Skipping batch {i // total_images_per_combined}, already processed.")
            continue

        # --- Step 4: Concatenate batch frames ---
        imgs = []
        for p in batch:
            img_path = os.path.join(input_folder, p)
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️ Skipping unreadable image: {p}")
                continue
            imgs.append(img)

        if not imgs:
            continue

        # Target cell dimensions
        cell_w = base_w
        cell_h = int(base_w / aspect_ratio)
        grid_w = images_per_row * cell_w
        grid_h = rows * cell_h
        combined = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

        # Pad if needed
        while len(imgs) < total_images_per_combined:
            imgs.append(np.zeros_like(sample_img))

        # --- Preserve aspect ratio during resizing ---
        for j, img in enumerate(imgs):
            r, c = divmod(j, images_per_row)
            h, w = img.shape[:2]
            scale = min(cell_w / w, cell_h / h)
            resized = cv2.resize(img, (int(w * scale), int(h * scale)))
            pad_x = (cell_w - resized.shape[1]) // 2
            pad_y = (cell_h - resized.shape[0]) // 2
            x1, y1 = c * cell_w + pad_x, r * cell_h + pad_y
            combined[y1:y1 + resized.shape[0], x1:x1 + resized.shape[1]] = resized

        combined_temp = os.path.join(output_folder, f"_combined_temp_{i:05d}.jpg")
        cv2.imwrite(combined_temp, combined)

        # --- Step 5: Run ComfyFlux workflow ---
        try:
            results = run_workflow_with_image(
                combined_temp,
                prompt_json_path=prompt_json_path,
                seed=seed,
                steps=steps,
                cfg=cfg,
                flux_guidance=flux_guidance,
                comfy_server=comfy_server,
                prompt_text=prompt_text
            )

            if not results:
                print(f"❌ No result for batch {i // total_images_per_combined}")
                continue

            colorized_path = results[0]
            colorized = cv2.imread(colorized_path)
            if colorized is None:
                print(f"❌ Failed to read output from ComfyFlux for batch {i}")
                continue

        except Exception as e:
            print(f"❌ Error in ComfyFlux for batch {i}: {e}")
            continue

        # --- Step 6: Split colorized grid back into frames ---
        h, w, _ = colorized.shape
        sub_w = w // images_per_row
        sub_h = h // rows

        frame_count = 0
        for r in range(rows):
            for c in range(images_per_row):
                if frame_count >= len(batch):
                    break
                y1, y2 = r * sub_h, (r + 1) * sub_h
                x1, x2 = c * sub_w, (c + 1) * sub_w
                sub_img = colorized[y1:y2, x1:x2]
                restored = cv2.resize(sub_img, (base_w, base_h))
                out_name = os.path.join(output_folder, f"frame_{i + frame_count:05d}.png")
                cv2.imwrite(out_name, restored)
                frame_count += 1

        print(f"✅ Batch {i // total_images_per_combined}: saved {frame_count} frames.")

        # --- Step 7: Cleanup temp file ---
        if os.path.exists(combined_temp):
            os.remove(combined_temp)

    print("🎉 All batches processed successfully!")









def process_scene_batches_without_overlap(
    input_folder,
    frames,
    output_folder,
    *,
    prompt_json_path=WORKFLOW_JSON,
    seed=2**24,
    steps=10,
    cfg=1.0,
    flux_guidance=2.5,
    comfy_server=COMFY,
    prompt_text=None,
    images_per_row=2,
    total_images_per_combined=6
):
    """
    Scene-wise batch colorization WITHOUT color overlap reuse.
    Each batch uses the next N grayscale frames directly (e.g., 0–5, 6–11, etc).
    Preserves aspect ratio and saves all frames per batch.
    """
    print_flag = False
    import os, cv2, math, numpy as np, gc, re
    from tqdm import tqdm
    from Utils.flux import run_workflow_with_image

    os.makedirs(output_folder, exist_ok=True)

    # --- ensure numeric sort ---
    def extract_scene_number(fname):
        match = re.search(r"Scene-(\d+)", fname)
        return int(match.group(1)) if match else 0

    frames = sorted(frames, key=extract_scene_number)

    # --- verify base ---
    sample_path = os.path.join(input_folder, frames[0])
    sample_img = cv2.imread(sample_path)
    if sample_img is None:
        print(f"⚠️ Cannot read base image: {sample_path}")
        return

    base_h, base_w = sample_img.shape[:2]
    aspect_ratio = base_w / base_h
    rows = math.ceil(total_images_per_combined / images_per_row)
    if(print_flag):
       print(f"[INFO] Scene: {os.path.basename(output_folder)} | Total frames: {len(frames)} | Grid {rows}x{images_per_row}")

    batch_idx = 0

    for i in range(0, len(frames), total_images_per_combined):
        batch_idx += 1
        batch_frames = frames[i:i + total_images_per_combined]
        if(print_flag):
           print(f"\n🟡 Building batch {batch_idx} | Frames {i}–{i + len(batch_frames) - 1}")

        imgs = []
        for f in batch_frames:
            img_path = os.path.join(input_folder, f)
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️ Missing {img_path}")
                continue
            imgs.append(img)

        # pad if needed
        while len(imgs) < total_images_per_combined:
            imgs.append(np.zeros_like(sample_img))
            print("     ➕ Added black filler frame to pad batch")

        combined_path = os.path.join(output_folder, f"_combined_{batch_idx:03d}.jpg")

        # --- build aspect-ratio-preserving grid ---
        cell_w, cell_h = base_w, int(base_w / aspect_ratio)
        grid_w, grid_h = images_per_row * cell_w, rows * cell_h
        combined = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

        for j, img in enumerate(imgs):
            if img is None:
                continue
            r, c = divmod(j, images_per_row)
            h, w = img.shape[:2]
            scale = min(cell_w / w, cell_h / h)
            resized = cv2.resize(img, (int(w * scale), int(h * scale)))
            pad_x = (cell_w - resized.shape[1]) // 2
            pad_y = (cell_h - resized.shape[0]) // 2
            x1, y1 = c * cell_w + pad_x, r * cell_h + pad_y
            combined[y1:y1 + resized.shape[0], x1:x1 + resized.shape[1]] = resized

        cv2.imwrite(combined_path, combined)
        if(print_flag):
           print(f"   🧩 Combined grid saved → {combined_path}")

        # --- Run ComfyFlux ---
        try:
            results = run_workflow_with_image(
                combined_path,
                prompt_json_path=prompt_json_path,
                seed=seed,
                steps=steps,
                cfg=cfg,
                flux_guidance=flux_guidance,
                comfy_server=comfy_server,
                prompt_text=prompt_text
            )

            if not results:
                print(f"❌ No result for batch {batch_idx}")
                continue

            colorized_img = cv2.imread(results[0])
            if colorized_img is None:
                print(f"❌ Cannot read colorized output for batch {batch_idx}")
                continue

            # --- Split colorized grid ---
            sub_w = colorized_img.shape[1] // images_per_row
            sub_h = colorized_img.shape[0] // rows
            split_colorized = []

            for r in range(rows):
                for c in range(images_per_row):
                    flat_idx = r * images_per_row + c
                    if flat_idx >= len(batch_frames):
                        break
                    y1, y2 = r * sub_h, (r + 1) * sub_h
                    x1, x2 = c * sub_w, (c + 1) * sub_w
                    sub = colorized_img[y1:y2, x1:x2]
                    restored = cv2.resize(sub, (base_w, base_h))
                    split_colorized.append(restored)

            # --- Save all colorized frames for this batch ---
            for j, colored in enumerate(split_colorized):
                out_idx = i + j
                if out_idx >= len(frames):
                    break
                out_path = os.path.join(output_folder, f"frame_{out_idx:05d}.png")
                cv2.imwrite(out_path, colored)
                if(print_flag):
                    print(f"     ✅ Saved → {out_path}")

        except Exception as e:
            print(f"❌ Error in batch {batch_idx}: {e}")

        gc.collect()

    if(print_flag):
       print(f"\n🎬 Completed scene {os.path.basename(output_folder)} | {batch_idx} batches processed.")








def process_scene_batches_without_overlap(
    input_folder,
    frames,
    output_folder,
    *,
    prompt_json_path=WORKFLOW_JSON,
    seed=2**24,
    steps=10,
    cfg=1.0,
    flux_guidance=2.5,
    comfy_server=COMFY,
    prompt_text=None,
    images_per_row=2,
    total_images_per_combined=6
):
    """
    Scene-wise batch colorization WITHOUT color overlap reuse.
    Each batch uses next N grayscale frames directly (0–5, 6–11, etc.).
    If fewer than required, duplicates last frame to fill grid.
    Preserves aspect ratio.
    """

    import os, cv2, math, numpy as np, gc, re
    from tqdm import tqdm
    from Utils.flux import run_workflow_with_image

    os.makedirs(output_folder, exist_ok=True)

    # --- ensure numeric sort ---
    def extract_scene_number(fname):
        match = re.search(r"Scene-(\d+)", fname)
        return int(match.group(1)) if match else 0

    frames = sorted(frames, key=extract_scene_number)

    # --- verify base ---
    sample_path = os.path.join(input_folder, frames[0])
    sample_img = cv2.imread(sample_path)
    if sample_img is None:
        print(f"⚠️ Cannot read base image: {sample_path}")
        return

    base_h, base_w = sample_img.shape[:2]
    aspect_ratio = base_w / base_h
    rows = math.ceil(total_images_per_combined / images_per_row)
    print(f"[INFO] Scene: {os.path.basename(output_folder)} | Total frames: {len(frames)} | Grid {rows}x{images_per_row}")

    batch_idx = 0

    for i in range(0, len(frames), total_images_per_combined):
        batch_idx += 1
        batch_frames = frames[i:i + total_images_per_combined]
        print(f"\n🟡 Building batch {batch_idx} | Frames {i}–{i + len(batch_frames) - 1}")

        imgs = []
        for f in batch_frames:
            img_path = os.path.join(input_folder, f)
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️ Missing {img_path}")
                continue
            imgs.append(img)

        # --- Duplicate last image if batch smaller than required ---
        if len(imgs) < total_images_per_combined:
            last_valid = imgs[-1] if imgs else np.zeros_like(sample_img)
            num_to_add = total_images_per_combined - len(imgs)
            for _ in range(num_to_add):
                imgs.append(last_valid.copy())
            print(f"     🌀 Duplicated last frame {num_to_add} times to fill grid")

        combined_path = os.path.join(output_folder, f"_combined_{batch_idx:03d}.jpg")

        # --- build aspect-ratio-preserving grid ---
        cell_w, cell_h = base_w, int(base_w / aspect_ratio)
        grid_w, grid_h = images_per_row * cell_w, rows * cell_h
        combined = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

        for j, img in enumerate(imgs):
            if img is None:
                continue
            r, c = divmod(j, images_per_row)
            h, w = img.shape[:2]
            scale = min(cell_w / w, cell_h / h)
            resized = cv2.resize(img, (int(w * scale), int(h * scale)))
            pad_x = (cell_w - resized.shape[1]) // 2
            pad_y = (cell_h - resized.shape[0]) // 2
            x1, y1 = c * cell_w + pad_x, r * cell_h + pad_y
            combined[y1:y1 + resized.shape[0], x1:x1 + resized.shape[1]] = resized

        cv2.imwrite(combined_path, combined)
        print(f"   🧩 Combined grid saved → {combined_path}")

        # --- Run ComfyFlux ---
        try:
            results = run_workflow_with_image(
                combined_path,
                prompt_json_path=prompt_json_path,
                seed=seed,
                steps=steps,
                cfg=cfg,
                flux_guidance=flux_guidance,
                comfy_server=comfy_server,
                prompt_text=prompt_text
            )

            if not results:
                print(f"❌ No result for batch {batch_idx}")
                continue

            colorized_img = cv2.imread(results[0])
            if colorized_img is None:
                print(f"❌ Cannot read colorized output for batch {batch_idx}")
                continue

            # --- Split colorized grid ---
            sub_w = colorized_img.shape[1] // images_per_row
            sub_h = colorized_img.shape[0] // rows
            split_colorized = []

            for r in range(rows):
                for c in range(images_per_row):
                    flat_idx = r * images_per_row + c
                    if flat_idx >= len(batch_frames):
                        break  # don't include duplicated fillers
                    y1, y2 = r * sub_h, (r + 1) * sub_h
                    x1, x2 = c * sub_w, (c + 1) * sub_w
                    sub = colorized_img[y1:y2, x1:x2]
                    restored = cv2.resize(sub, (base_w, base_h))
                    split_colorized.append(restored)

            # --- Save actual (non-duplicate) frames only ---
            for j, colored in enumerate(split_colorized):
                out_idx = i + j
                if out_idx >= len(frames):
                    break
                out_path = os.path.join(output_folder, f"frame_{out_idx:05d}.png")
                cv2.imwrite(out_path, colored)
                print(f"     ✅ Saved → {out_path}")

        except Exception as e:
            print(f"❌ Error in batch {batch_idx}: {e}")

        gc.collect()

    print(f"\n🎬 Completed scene {os.path.basename(output_folder)} | {batch_idx} batches processed.")




def process_scene_batches_without_overlap(
    input_folder,
    frames,
    output_folder,
    *,
    prompt_json_path=WORKFLOW_JSON,
    seed=2**24,
    steps=10,
    cfg=1.0,
    flux_guidance=2.5,
    comfy_server=COMFY,
    prompt_text=None,
    images_per_row=2,
    total_images_per_combined=6
):
    """
    Scene-wise batch colorization WITHOUT overlap reuse.
    Adds caching — skips already processed batches.
    - Duplicates last frame if fewer than total_images_per_combined.
    - Preserves aspect ratio.
    - Saves combined input + colorized combined grids.
    """

    import os, cv2, math, numpy as np, gc, re
    from tqdm import tqdm
    from Utils.flux import run_workflow_with_image

    os.makedirs(output_folder, exist_ok=True)
    print_flag =False
    # --- ensure numeric sort ---
    def extract_scene_number(fname):
        match = re.search(r"Scene-(\d+)", fname)
        return int(match.group(1)) if match else 0

    frames = sorted(frames, key=extract_scene_number)

    # --- verify base ---
    sample_path = os.path.join(input_folder, frames[0])
    sample_img = cv2.imread(sample_path)
    if sample_img is None:
        print(f"⚠️ Cannot read base image: {sample_path}")
        return

    base_h, base_w = sample_img.shape[:2]
    aspect_ratio = base_w / base_h
    rows = math.ceil(total_images_per_combined / images_per_row)
    if(print_flag):
       print(f"[INFO] Scene: {os.path.basename(output_folder)} | Total frames: {len(frames)} | Grid {rows}x{images_per_row}")

    batch_idx = 0

    for i in range(0, len(frames), total_images_per_combined):
        batch_idx += 1
        batch_frames = frames[i:i + total_images_per_combined]

        combined_path = os.path.join(output_folder, f"_combined_{batch_idx:03d}.jpg")
        colorized_combined_path = os.path.join(output_folder, f"_combined_colorized_{batch_idx:03d}.jpg")

        # --- Check cache ---
        frame_exists = all(
            os.path.exists(os.path.join(output_folder, f"frame_{i+j:05d}.png"))
            for j in range(len(batch_frames))
        )
        if os.path.exists(colorized_combined_path) and frame_exists:
            print(f"[CACHE] Skipping batch {batch_idx} (already colorized & frames exist)")
            continue

        if(print_flag):
           print(f"\n🟡 Building batch {batch_idx} | Frames {i}–{i + len(batch_frames) - 1}")

        imgs = []
        for f in batch_frames:
            img_path = os.path.join(input_folder, f)
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️ Missing {img_path}")
                continue
            imgs.append(img)

        # --- Duplicate last image if fewer than required ---
        if len(imgs) < total_images_per_combined:
            last_valid = imgs[-1] if imgs else np.zeros_like(sample_img)
            num_to_add = total_images_per_combined - len(imgs)
            for _ in range(num_to_add):
                imgs.append(last_valid.copy())
            if(print_flag):
               print(f"     🌀 Duplicated last frame {num_to_add}× to fill grid")

        # --- Build combined grid ---
        cell_w, cell_h = base_w, int(base_w / aspect_ratio)
        grid_w, grid_h = images_per_row * cell_w, rows * cell_h
        combined = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

        for j, img in enumerate(imgs):
            if img is None:
                continue
            r, c = divmod(j, images_per_row)
            h, w = img.shape[:2]
            scale = min(cell_w / w, cell_h / h)
            resized = cv2.resize(img, (int(w * scale), int(h * scale)))
            pad_x = (cell_w - resized.shape[1]) // 2
            pad_y = (cell_h - resized.shape[0]) // 2
            x1, y1 = c * cell_w + pad_x, r * cell_h + pad_y
            combined[y1:y1 + resized.shape[0], x1:x1 + resized.shape[1]] = resized

        # --- Save combined frame ---
        if not os.path.exists(combined_path):
            cv2.imwrite(combined_path, combined)
            if(print_flag):
                print(f"   🧩 Combined grid saved → {combined_path}")
        else:
            if(print_flag):
                print(f"   [CACHE] Combined grid exists → {combined_path}")

        # --- Run ComfyFlux only if colorized output missing ---
        if not os.path.exists(colorized_combined_path):
            if(print_flag):
                print("   🚀 Running ComfyFlux pipeline...")
            try:
                results = run_workflow_with_image(
                    combined_path,
                    prompt_json_path=prompt_json_path,
                    seed=seed,
                    steps=steps,
                    cfg=cfg,
                    flux_guidance=flux_guidance,
                    comfy_server=comfy_server,
                    prompt_text=prompt_text
                )

                if not results:
                    print(f"❌ No result for batch {batch_idx}")
                    continue

                colorized_img = cv2.imread(results[0])
                if colorized_img is None:
                    print(f"❌ Cannot read colorized output for batch {batch_idx}")
                    continue

                cv2.imwrite(colorized_combined_path, colorized_img)
                if(print_flag):
                    print(f"   🌈 Colorized combined saved → {colorized_combined_path}")

            except Exception as e:
                print(f"❌ Error in ComfyFlux for batch {batch_idx}: {e}")
                continue
        else:
            if(print_flag):
                print(f"   [CACHE] Using cached colorized → {colorized_combined_path}")
            colorized_img = cv2.imread(colorized_combined_path)

        # --- Split colorized grid ---
        sub_w = colorized_img.shape[1] // images_per_row
        sub_h = colorized_img.shape[0] // rows
        split_colorized = []

        for r in range(rows):
            for c in range(images_per_row):
                flat_idx = r * images_per_row + c
                if flat_idx >= len(batch_frames):
                    break  # skip duplicates
                y1, y2 = r * sub_h, (r + 1) * sub_h
                x1, x2 = c * sub_w, (c + 1) * sub_w
                sub = colorized_img[y1:y2, x1:x2]
                restored = cv2.resize(sub, (base_w, base_h))
                split_colorized.append(restored)

        # --- Save real (non-duplicated) frames ---
        for j, colored in enumerate(split_colorized):
            out_idx = i + j
            out_path = os.path.join(output_folder, f"frame_{out_idx:05d}.png")
            if os.path.exists(out_path):
                continue  # cached frame
            cv2.imwrite(out_path, colored)
            if(print_flag):
                print(f"     ✅ Saved → {out_path}")

        gc.collect()

    if(print_flag):
        print(f"\n🎬 Completed scene {os.path.basename(output_folder)} | {batch_idx} batches processed.")



import re

def process_scene_batches(
    input_folder,
    frames,
    output_folder,
    *,
    prompt_json_path=WORKFLOW_JSON,
    seed=2**24,
    steps=10,
    cfg=1.0,
    flux_guidance=2.5,
    comfy_server=COMFY,
    prompt_text=None,
    images_per_row=2,
    total_images_per_combined=6,
    overlap_frames=2
):
    """
    Scene-wise sliding batch colorization with overlap continuity.
    ✅ Fixes off-by-two frame index drift in saved outputs.
    """

    import os, cv2, math, numpy as np, gc, re
    from tqdm import tqdm
    from Utils.flux import run_workflow_with_image

    os.makedirs(output_folder, exist_ok=True)
    last_colorized = []

    # --- ensure numeric sort ---
    def extract_scene_number(fname):
        match = re.search(r"Scene-(\d+)", fname)
        return int(match.group(1)) if match else 0

    frames = sorted(frames, key=extract_scene_number)

    # --- verify base ---
    sample_path = os.path.join(input_folder, frames[0])
    sample_img = cv2.imread(sample_path)
    if sample_img is None:
        print(f"⚠️ Cannot read base image: {sample_path}")
        return

    base_h, base_w = sample_img.shape[:2]
    aspect_ratio = base_w / base_h
    rows = math.ceil(total_images_per_combined / images_per_row)
    print(f"[INFO] Scene: {os.path.basename(output_folder)} | Total frames: {len(frames)} | Grid {rows}x{images_per_row}")

    idx = 0
    batch_idx = 0

    while idx < len(frames):
        batch_idx += 1
        is_first_batch = (batch_idx == 1)
        batch_inputs = []

        print(f"\n🟡 Building batch {batch_idx} | idx={idx}")

        # overlap reuse
        if last_colorized:
            print(f"   🔁 Using {len(last_colorized)} overlap colorized frames from previous batch.")
            for n, prev_colored in enumerate(last_colorized):
                batch_inputs.append(prev_colored)
                print(f"     ↳ [Overlap #{n+1}] (colorized array)")
        else:
            print("   🆕 No overlap (first batch).")

        # corrected grayscale selection
        start_idx = idx + (0 if is_first_batch else overlap_frames)
        end_idx = start_idx + (total_images_per_combined - len(last_colorized))
        next_bw_frames = frames[start_idx:end_idx]

        print(f"   ➕ Adding grayscale frames from indices {start_idx}–{end_idx-1}:")
        for f in next_bw_frames:
            img_path = os.path.join(input_folder, f)
            print(f"     • {img_path}")
            img = cv2.imread(img_path)
            if img is None:
                print(f"       ⚠️ Missing or unreadable: {img_path}")
                continue
            batch_inputs.append(img)

        while len(batch_inputs) < total_images_per_combined:
            batch_inputs.append(np.zeros_like(sample_img))
            print("     ➕ Added black filler frame to pad batch")

        combined_path = os.path.join(output_folder, f"_combined_{batch_idx:03d}.jpg")

        # build grid
        cell_w, cell_h = base_w, int(base_w / aspect_ratio)
        grid_w, grid_h = images_per_row * cell_w, rows * cell_h
        combined = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

        for j, img in enumerate(batch_inputs):
            if img is None:
                continue
            r, c = divmod(j, images_per_row)
            h, w = img.shape[:2]
            scale = min(cell_w / w, cell_h / h)
            resized = cv2.resize(img, (int(w * scale), int(h * scale)))
            pad_x = (cell_w - resized.shape[1]) // 2
            pad_y = (cell_h - resized.shape[0]) // 2
            x1, y1 = c * cell_w + pad_x, r * cell_h + pad_y
            combined[y1:y1 + resized.shape[0], x1:x1 + resized.shape[1]] = resized

        cv2.imwrite(combined_path, combined)
        print(f"   🧩 Combined grid saved → {combined_path}")

        # run comfyflux
        try:
            results = run_workflow_with_image(
                combined_path,
                prompt_json_path=prompt_json_path,
                seed=seed,
                steps=steps,
                cfg=cfg,
                flux_guidance=flux_guidance,
                comfy_server=comfy_server,
                prompt_text=prompt_text
            )

            if not results:
                print(f"❌ No result for batch {batch_idx}")
                idx += (total_images_per_combined - overlap_frames)
                continue

            colorized_img = cv2.imread(results[0])
            if colorized_img is None:
                print(f"❌ Cannot read colorized output for batch {batch_idx}")
                idx += (total_images_per_combined - overlap_frames)
                continue

            # split grid
            sub_w = colorized_img.shape[1] // images_per_row
            sub_h = colorized_img.shape[0] // rows
            split_colorized = []

            for r in range(rows):
                for c in range(images_per_row):
                    flat_idx = r * images_per_row + c
                    if flat_idx >= len(batch_inputs):
                        break
                    y1, y2 = r * sub_h, (r + 1) * sub_h
                    x1, x2 = c * sub_w, (c + 1) * sub_w
                    sub = colorized_img[y1:y2, x1:x2]
                    restored = cv2.resize(sub, (base_w, base_h))
                    split_colorized.append(restored)

            # save frames
            if is_first_batch:
                save_start = 0
            else:
                save_start = overlap_frames

            print(f"   💾 Saving frames {save_start}–{len(split_colorized)-1} (out_idx starts from {idx})")
            for i_save in range(save_start, len(split_colorized)):
                out_idx = idx + i_save
                if out_idx >= len(frames):
                    break
                out_name = os.path.join(output_folder, f"frame_{out_idx:05d}.png")
                cv2.imwrite(out_name, split_colorized[i_save])
                print(f"     ✅ Saved → {out_name}")

            last_colorized = split_colorized[-overlap_frames:]
            print(f"   🔁 Updated overlap frames → {len(last_colorized)} frames carried forward.")

        except Exception as e:
            print(f"❌ Error in batch {batch_idx}: {e}")

        # move forward correctly
        idx += (total_images_per_combined - overlap_frames)
        print(f"   🔚 Moving idx to {idx} for next batch.\n")
        gc.collect()

    print(f"\n🎬 Completed scene {os.path.basename(output_folder)} | {batch_idx} batches processed.")


# --- Example call ---
# Put any image path you like below; this is the "dynamic" input.
# dynamic_input = "output/input_2.png"  # replace with your image
# prompt_text = "restore and colorize this, no warm/cool tint in entire image, color background, natural skintones, each person's dress differently (different color, natural, light, vivid)" 
# prompt_text = "restore and colorize this, no warm/cool tint in entire image, color background, natural skintones"
# results = run_workflow_with_image(dynamic_input, prompt_text = prompt_text)
# print("Saved outputs:")
# for p in results:
#     print(" -", p)

# # Display first image inline (optional)
# from IPython.display import display
# display(Image.open(results[0]))
