import os, json, uuid, time, requests
from PIL import Image
from tqdm import tqdm

COMFY = "http://127.0.0.1:8188"
PROMPT_JSON = "image_qwen_image_edit.json"
OUTPUT_DIR = "results_qwen_api"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Upload image to ComfyUI ---
def upload_image(image_path, server=COMFY):
    img_name = os.path.basename(image_path)
    with open(image_path, "rb") as f:
        files = {"image": (img_name, f, "image/png")}
        data = {"type": "input", "overwrite": "true"}
        r = requests.post(f"{server}/upload/image", files=files, data=data, timeout=120)
        r.raise_for_status()
    return img_name


# --- Download result image ---
def download_image(filename, server=COMFY, *, subfolder="", folder_type="output"):
    params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    r = requests.get(f"{server}/view", params=params, timeout=120)
    r.raise_for_status()
    local_path = os.path.join(OUTPUT_DIR, filename)
    with open(local_path, "wb") as f:
        f.write(r.content)
    return local_path


# --- Run ComfyUI prompt with adjustable parameters ---
def run_prompt(prompt_json_path, image_path, new_prompt, server=COMFY, seed=None, steps=None, cfg=None):
    print_flag =False
    with open(prompt_json_path, "r") as f:
        prompt_data = json.load(f)

    # Upload image
    uploaded_img = upload_image(image_path, server)
    if(print_flag):
        print(f"✅ Uploaded: {uploaded_img}")

    # Patch LoadImage node
    if "78" in prompt_data:
        prompt_data["78"]["inputs"]["image"] = uploaded_img

    # Patch positive prompt
    if "76" in prompt_data:
        prompt_data["76"]["inputs"]["prompt"] = new_prompt

    # Clear negative prompt
    if "77" in prompt_data:
        prompt_data["77"]["inputs"]["prompt"] = "blue clothing, blue outfit, blue dress"

    # Patch KSampler parameters
    if "3" in prompt_data:
        ks = prompt_data["3"]["inputs"]
        if seed is not None:
            ks["seed"] = int(seed)
        if steps is not None:
            ks["steps"] = int(steps)
        if cfg is not None:
            ks["cfg"] = float(cfg)

    # Fix SaveImage node inputs
    if "60" in prompt_data:
        prompt_data["60"]["inputs"] = {
            "filename_prefix": "ComfyUI",
            "images": ["8", 0]
        }

    # Submit prompt
    client_id = str(uuid.uuid4())
    payload = {"prompt": prompt_data, "client_id": client_id}
    r = requests.post(f"{server}/prompt", json=payload, timeout=120)

    if r.status_code != 200:
        print("❌ ComfyUI rejected the request:")
        print("Status:", r.status_code)
        print("Response:", r.text)
        return None

    data = r.json()
    prompt_id = data.get("prompt_id", client_id)
    if(print_flag):
       print(f"✅ Queued prompt: {prompt_id}")

    # Poll for output
    deadline = time.time() + 6000
    while time.time() < deadline:
        hist = requests.get(f"{server}/history/{prompt_id}", timeout=60).json()
        item = hist.get(prompt_id)
        if item and "outputs" in item:
            for node_id, node_out in item["outputs"].items():
                for im in node_out.get("images", []):
                    fn = im.get("filename")
                    sub = im.get("subfolder", "")
                    typ = im.get("type", "output")
                    if fn:
                        out = download_image(fn, server, subfolder=sub, folder_type=typ)
                        if(print_flag):
                           print(f"✅ Saved: {out}")
                        return out
        time.sleep(0.5)

    print("❌ Timeout: No output image found.")
    return None


# --- Batch processor with tqdm progress ---
def process_folder(input_folder, output_folder, prompt_text, seed=None, steps=None, cfg=None):
    print_flag =False
    os.makedirs(output_folder, exist_ok=True)
    image_files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    print(f"\n📂 Found {len(image_files)} image(s) to process.\n")
    success_count, skip_count, fail_count = 0, 0, 0

    for fname in tqdm(image_files, desc="Processing images", ncols=100):
        in_path = os.path.join(input_folder, fname)
        out_path = os.path.join(output_folder, fname)

        if os.path.exists(out_path):
            skip_count += 1
            if(print_flag):
                tqdm.write(f"⚙️  Skipping {fname}, already exists.")
            continue

        try:
            if(print_flag):
                tqdm.write(f"🚀 Processing {fname} ...")
            result = run_prompt(PROMPT_JSON, in_path, prompt_text, seed=seed, steps=steps, cfg=cfg)

            if result:
                orig_w, orig_h = Image.open(in_path).size
                out_img = Image.open(result)
                out_img = out_img.resize((orig_w, orig_h), Image.LANCZOS)
                out_img.save(out_path)
                success_count += 1
                if(print_flag):
                    tqdm.write(f"✅ Saved → {out_path} ({orig_w}x{orig_h})")
            else:
                fail_count += 1
                if(print_flag):
                    tqdm.write(f"⚠️ No output for {fname}")

        except Exception as e:
            fail_count += 1
            tqdm.write(f"❌ Error processing {fname}: {e}")

        time.sleep(1)

    print("\n📊 Summary:")
    print(f"✅ Success: {success_count}")
    print(f"⚙️  Skipped: {skip_count}")
    print(f"❌ Failed:  {fail_count}")
    print(f"📁 Output folder: {output_folder}")


# # ------------------- Example Usage -------------------
# if __name__ == "__main__":
#     input_folder = "input/images"
#     output_folder = "output/results_colorized_5_1.5"

#     prompt_text = (
#         "restore and colorize with vivid colors, "
#         "skin tones should be natural, realistic, "
#         "and no warm or cool tint in the entire image"
#     )

#     process_folder(
#         input_folder,
#         output_folder,
#         prompt_text,
#         seed=123456789,   # 👈 your seed
#         steps=5,         # 👈 number of denoising steps
#         cfg=1.5           # 👈 CFG guidance
#     )
