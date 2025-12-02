import argparse
import os
import glob
from pathlib import Path
import cv2
from tqdm import tqdm
import torch
import numpy as np

# --- GFPGAN imports ---
from gfpgan import GFPGANer
# --- Real-ESRGAN imports ---
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


def process_frame(frame, gfpgan_restorer=None, realesrgan_upsampler=None,
                  weight=0.5, use_gfpgan=True, use_realesrgan=True):
    """Process a single frame/image and ensure it’s valid for saving."""
    out_frame = frame.copy()

    # --- GFPGAN restoration ---
    if use_gfpgan and gfpgan_restorer is not None:
        try:
            _, _, out_frame = gfpgan_restorer.enhance(
                out_frame, has_aligned=False, only_center_face=False,
                paste_back=True, weight=weight
            )
        except Exception as e:
            print(f"[WARN] GFPGAN failed on frame: {e}")
            out_frame = frame

    # --- Real-ESRGAN upscaling ---
    if use_realesrgan and realesrgan_upsampler is not None:
        try:
            out_frame, _ = realesrgan_upsampler.enhance(out_frame, outscale=2)
        except Exception as e:
            print(f"[WARN] Real-ESRGAN failed on frame: {e}")
            out_frame = frame

    # --- Ensure proper dtype and shape ---
    if out_frame.dtype != np.uint8:
        out_frame = np.clip(out_frame, 0, 255).astype(np.uint8)
    if len(out_frame.shape) == 2:  # grayscale → 3-channel
        out_frame = cv2.cvtColor(out_frame, cv2.COLOR_GRAY2BGR)

    return out_frame


import contextlib
import io

def process_frame(frame, gfpgan_restorer=None, realesrgan_upsampler=None, weight=0.5, use_gfpgan=True, use_realesrgan=True):
    """Process a single frame or image while suppressing Real-ESRGAN/GFPGAN tile prints."""
    out_frame = frame.copy()

    if use_gfpgan and gfpgan_restorer is not None:
        # suppress GFPGAN prints
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            _, _, out_frame = gfpgan_restorer.enhance(
                out_frame, has_aligned=False, only_center_face=False, paste_back=True, weight=weight
            )

    if use_realesrgan and realesrgan_upsampler is not None:
        # suppress Real-ESRGAN tile prints
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            out_frame, _ = realesrgan_upsampler.enhance(out_frame, outscale=2)

    # Ensure uint8 for VideoWriter
    if out_frame.dtype != np.uint8:
        out_frame = np.clip(out_frame, 0, 255).astype(np.uint8)

    return out_frame



def process_images(input_path, output_path, gfpgan_restorer, realesrgan_upsampler,
                   weight, use_gfpgan, use_realesrgan, ext='auto', suffix=None):
    """Process a folder or single image."""
    if os.path.isfile(input_path):
        img_list = [input_path]
    else:
        img_list = sorted(glob.glob(os.path.join(input_path, '*')))
    os.makedirs(output_path, exist_ok=True)

    for img_path in tqdm(img_list, desc="Processing images"):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[WARN] Failed to read: {img_path}")
            continue

        out_img = process_frame(img, gfpgan_restorer, realesrgan_upsampler,
                                weight, use_gfpgan, use_realesrgan)

        basename, orig_ext = os.path.splitext(os.path.basename(img_path))
        save_ext = orig_ext[1:] if ext == 'auto' else ext
        out_file = os.path.join(
            output_path,
            f"{basename}_{suffix}.{save_ext}" if suffix else f"{basename}.{save_ext}"
        )

        success = cv2.imwrite(out_file, out_img)
        if not success:
            print(f"[ERROR] Could not write image: {out_file}")
        else:
            print(f"[OK] Saved: {out_file}")


def process_video(input_video, output_video, gfpgan_restorer, realesrgan_upsampler,
                  weight, use_gfpgan, use_realesrgan):
    """Process video frame-by-frame with automatic codec fallback."""
    Path(output_video).parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # --- Try codecs in order: mp4v -> XVID -> avc1 ---
    codec_list = ['mp4v', 'XVID', 'avc1']
    out = None
    for codec in codec_list:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        if out.isOpened():
            print(f"[INFO] Using codec: {codec}")
            break

    if not out or not out.isOpened():
        raise RuntimeError("Failed to open VideoWriter! "
                           "None of the codecs (mp4v, XVID, avc1) worked. "
                           "Try changing output extension or installing FFmpeg with codec support.")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=frame_count, desc="Processing video frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        out_frame = process_frame(frame, gfpgan_restorer, realesrgan_upsampler,
                                  weight, use_gfpgan, use_realesrgan)

        # Resize if Real-ESRGAN changed the size
        if out_frame.shape[1] != width or out_frame.shape[0] != height:
            out_frame = cv2.resize(out_frame, (width, height))

        out.write(out_frame)
        pbar.update(1)

    cap.release()
    out.release()
    pbar.close()
    print(f"[OK] Video saved at: {output_video}")
    return output_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Input image/video or folder')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Output folder or video path')
    parser.add_argument('--use_gfpgan', action='store_true',
                        help='Enable GFPGAN face restoration')
    parser.add_argument('--gfpgan_version', type=str, default='1.3',
                        help='GFPGAN version: 1 | 1.2 | 1.3')
    parser.add_argument('--use_realesrgan', action='store_true',
                        help='Enable Real-ESRGAN upscaling')
    parser.add_argument('--weight', type=float, default=0.5,
                        help='GFPGAN enhancement weight')
    parser.add_argument('--bg_tile', type=int, default=400,
                        help='Tile size for Real-ESRGAN')
    parser.add_argument('--suffix', type=str, default=None,
                        help='Suffix for output images')
    parser.add_argument('--ext', type=str, default='auto',
                        help='Output image extension')
    args = parser.parse_args()

    # --- GFPGAN setup ---
    if args.use_gfpgan:
        if args.gfpgan_version == '1':
            arch, ch_mul, model_name, url = 'original', 1, 'GFPGANv1', \
                'https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth'
        elif args.gfpgan_version == '1.2':
            arch, ch_mul, model_name, url = 'clean', 2, 'GFPGANCleanv1-NoCE-C2', \
                'https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth'
        else:
            arch, ch_mul, model_name, url = 'clean', 2, 'GFPGANv1.3', \
                'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'

        model_path = os.path.join('experiments/pretrained_models', model_name + '.pth')
        if not os.path.isfile(model_path):
            model_path = url
        gfpgan_restorer = GFPGANer(model_path=model_path, upscale=2,
                                   arch=arch, channel_multiplier=ch_mul)
    else:
        gfpgan_restorer = None

    # --- Real-ESRGAN setup ---
    if args.use_realesrgan:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=2)
        realesrgan_upsampler = RealESRGANer(
            scale=2,
            model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
            model=model,
            tile=args.bg_tile, tile_pad=10, pre_pad=0, half=True
        )
    else:
        realesrgan_upsampler = None

    # --- Detect video or image input ---
    if os.path.isfile(args.input) and args.input.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
        process_video(args.input, args.output, gfpgan_restorer, realesrgan_upsampler,
                      args.weight, args.use_gfpgan, args.use_realesrgan)
    else:
        process_images(args.input, args.output, gfpgan_restorer, realesrgan_upsampler,
                       args.weight, args.use_gfpgan, args.use_realesrgan, args.ext, args.suffix)

    print(f"✅ Processing complete. Output at: {args.output}")


if __name__ == '__main__':
    main()
