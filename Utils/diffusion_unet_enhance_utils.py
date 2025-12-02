import os
import time
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from diffusers import (
    ControlNetModel,
    DDIMScheduler,
    StableDiffusionControlNetImg2ImgPipeline,
)
from diffusers.models import AutoencoderKL

# === Config ===
MODELS_ROOT = "./models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STRENGTH = 0.5
NUM_INFERENCE_STEPS = 10
GUIDANCE_SCALE = 10
CONTROLNET_SCALE = 0.85
SEED = 97


# === Timing decorator ===
def timer_func(func):
    def wrapper(*args, **kwargs):
        t0 = time.time()
        out = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time()-t0:.2f}s")
        return out
    return wrapper


# === Pipeline loader ===
class LazyLoadPipeline:
    def __init__(self):
        self.pipe = None

    @timer_func
    def load(self):
        if self.pipe is None:
            # ControlNet
            cn = ControlNetModel.from_single_file(
                os.path.join(MODELS_ROOT, "models", "ControlNet", "control_v11f1e_sd15_tile.pth"),
                torch_dtype=torch.float16,
            )
            # Stable Diffusion ControlNet pipeline
            self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_single_file(
                os.path.join(MODELS_ROOT, "models", "Stable-diffusion", "juggernaut_reborn.safetensors"),
                controlnet=cn,
                torch_dtype=torch.float16,
                use_safetensors=True,
                safety_checker=None,
            )
            # VAE
            vae = AutoencoderKL.from_single_file(
                os.path.join(MODELS_ROOT, "models", "VAE", "vae-ft-mse-840000-ema-pruned.safetensors"),
                torch_dtype=torch.float16,
            )
            self.pipe.vae = vae

            # Textual inversions
            self.pipe.load_textual_inversion(
                os.path.join(MODELS_ROOT, "models", "embeddings", "verybadimagenegative_v1.3.pt")
            )
            self.pipe.load_textual_inversion(
                os.path.join(MODELS_ROOT, "models", "embeddings", "JuggernautNegative-neg.pt")
            )

            # LoRA
            l1 = os.path.join(MODELS_ROOT, "models", "Lora", "SDXLrender_v2.0.safetensors")
            l2 = os.path.join(MODELS_ROOT, "models", "Lora", "more_details.safetensors")
            self.pipe.load_lora_weights(l1)
            self.pipe.fuse_lora(lora_scale=0.5)
            self.pipe.load_lora_weights(l2)
            self.pipe.fuse_lora(lora_scale=1.0)

            # Scheduler
            self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

            # Move to device
            self.pipe.to(DEVICE)

    def set_ddim(self):
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

    def __call__(self, **kwargs):
        return self.pipe(**kwargs)


# === Main function: video-to-video enhancement ===
@timer_func
def diffusion_unet_enhance(input_path, output_path):
    lazy_pipe = LazyLoadPipeline()
    lazy_pipe.load()
    lazy_pipe.set_ddim()

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_vid = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_count = 0
    with tqdm(total=total_frames, desc="Processing frames", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

            if pil_img.width >= 600:
                pil_img = pil_img.resize((pil_img.width // 2, pil_img.height // 2))

            result = lazy_pipe(
                prompt=(
                    "smooth, consistent vibrant clothes; ABSOLUTELY no blotches or inconsistent colorbleeds on fabrics; "
                    "lush green, bright blue skies, serene blue waters in the background; accentuated colors, walls, and furniture;"
                ),
                negative_prompt=(
                    "grayscale, rust colors, spotty color patches, spotty blotches, inconsistent, modernistic fashion, colored hair"
                ),
                image=pil_img,
                control_image=pil_img,
                num_inference_steps=NUM_INFERENCE_STEPS,
                strength=STRENGTH,
                guidance_scale=GUIDANCE_SCALE,
                controlnet_conditioning_scale=CONTROLNET_SCALE,
                generator=torch.Generator(device=DEVICE).manual_seed(SEED + frame_count),
            ).images[0]

            result_bgr = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
            result_resized = cv2.resize(result_bgr, (width, height))
            out_vid.write(result_resized)

            frame_count += 1
            pbar.update(1)

    cap.release()
    out_vid.release()
    print("Video processing complete.")


# === Example usage ===
# diffusion_unet_enhance("input_videos/input.mp4", "input_videos/output_enhanced.mp4")
