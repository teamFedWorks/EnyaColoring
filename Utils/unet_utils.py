import os
import warnings
from deoldify import device
from deoldify.device_id import DeviceId
device.set(device=DeviceId.GPU0)
import torch
import fastai
from deoldify.visualize import *
import torchvision
import tensorflow as tf
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from Utils.unet_model import ColoringModel
import hashlib
import itertools
import random
import re
import subprocess
import tempfile
import traceback
import pandas as pd
from skimage.color import lab2rgb, rgb2lab
from sklearn.cluster import KMeans
from tensorflow.keras.layers import (
    Add,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dense,
    GlobalAveragePooling2D,
    Input,
    Lambda,
    Layer,
    Multiply,
    Reshape,
    UpSampling2D,
)
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Sequence
from tqdm import tqdm
from transformers import (
    DPTFeatureExtractor,
    DPTForDepthEstimation,
    DPTImageProcessor,
    SamModel,
    SamProcessor,
)

import shutil
import sys
import gc
from Utils.main_utils import get_cached_file

print(tf.__version__)
print(tf.test.is_gpu_available())

warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")

# Initialize the colorizer
colorizer = get_image_colorizer(artistic=True)
def deoldify_inference(frame_rgb):
    ret = colorizer.get_transformed_image(frame_rgb, render_factor=16, post_process=True)
    return np.array(ret)  # Ensure NumPy array is returned
    
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())

torch.cuda.synchronize()  # Ensure all GPU ops finish before clearing memory
torch.cuda.empty_cache()

sam2_predictor, processor, grounding_model, TEXT_PROMPT = None,None,None,None

labels = ["dress", "nature", "wall", "sky", "ground", "light", "person", "furniture", "face"]

"""
Hyper parameters
"""
if(True):
    def generate_text_prompt(labels):
        return ". ".join(labels) + "."
    
    labels = ["dress", "nature", "wall", "sky", "ground", "light", "person", "furniture", "face"]
          
    total_labels_count = len(labels)
    
    TEXT_PROMPT = generate_text_prompt(labels)
    
    GROUNDING_MODEL = "IDEA-Research/grounding-dino-tiny"
   
    SAM2_CHECKPOINT = "models/sam2.1_hiera_large.pt"
    SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    OUTPUT_DIR = Path("output")
    # create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()
    
    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # build SAM2 image predictor
    sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    
    # build grounding dino from huggingface
    model_id = GROUNDING_MODEL
    processor = AutoProcessor.from_pretrained(model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(DEVICE)

def custom_get_phrases_from_posmap(posmaps, input_ids, probs):
  #posmaps = posmaps.clone()
  left_idx = 0
  right_idx = posmaps.shape[-1] - 1
  # Avoiding altering the input tensor
  posmaps = posmaps.clone()

  posmaps[:, 0 : left_idx + 1] = False
  posmaps[:, right_idx:] = False

  token_ids = []
  for i,posmap in enumerate(posmaps):
      non_zero_idx = posmap.nonzero(as_tuple=True)[0].tolist()
      if len(non_zero_idx) > 0:
          # Get the index with the highest probability
          max_idx = torch.argmax(probs[i]).item()
          token_ids.append([input_ids[max_idx]])
  return token_ids

def custom_output_labels(outputs,input_ids, threshold=0.12,text_threshold=0.12):
        batch_logits, batch_boxes = outputs.logits, outputs.pred_boxes
        batch_probs = torch.sigmoid(batch_logits)  # (batch_size, num_queries, 256)
        batch_scores = torch.max(batch_probs, dim=-1)[0]  # (batch_size, num_queries)
        labels_list =[]
        for idx, (scores, boxes, probs) in enumerate(zip(batch_scores, batch_boxes, batch_probs)):
            keep = scores > threshold
            prob = probs[keep]
            label_ids = custom_get_phrases_from_posmap(prob > text_threshold, input_ids[idx],prob)
            labels = processor.batch_decode(label_ids)
            labels_list.append(labels)
        return labels_list

def extract_class_masks(image, sam2_predictor, processor, grounding_model, TEXT_PROMPT, DEVICE, labels_list):
    """
    Ultra-optimized extraction of masks per class using Grounding DINO and SAM2.
    Uses PyTorch tensors and vectorized confidence-weighted mask aggregation.

    Args:
        image (PIL.Image): Input image.
        sam2_predictor: SAM2 model predictor.
        processor: Grounding DINO processor.
        grounding_model: Grounding DINO model.
        TEXT_PROMPT (str): The text prompt for object detection.
        DEVICE (str): The device (CPU/GPU) to run the model on.
        labels_list (list): List of expected label names in the desired order.

    Returns:
        ordered_masks (dict): Dictionary of label names mapped to their aggregated confidence masks (in order).
    """
    # Convert image format
    image = image.convert("L").convert("RGB")
    image_np = np.array(image)
    sam2_predictor.set_image(image_np)

    # Process input with Grounding DINO
    inputs = processor(images=image, text=TEXT_PROMPT, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    # Get detection results
    results = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, box_threshold=0.12, text_threshold=0.12, target_sizes=[image.size[::-1]]
    )

    input_boxes = results[0]["boxes"].to(DEVICE)
    confidences = results[0]["scores"].to(DEVICE)
    concatenated_labels = results[0]["labels"]
    concatenated_confidences = results[0]["scores"].cpu().numpy().tolist()
    detected_labels = custom_output_labels(outputs, inputs.input_ids, threshold=0.12, text_threshold=0.12)[0]

    # Predict masks using SAM2
    if input_boxes.shape[0] > 0:
        masks, _, _ = sam2_predictor.predict(
            point_coords=None, point_labels=None, box=input_boxes.cpu().numpy(), multimask_output=False
        )
        # Safely squeeze only if the second dimension exists
        if masks.ndim == 4:
            masks = masks.squeeze(1)  # Remove unnecessary dimensions
        masks = torch.from_numpy(masks).to(dtype=torch.float32, device=DEVICE)
    else:
        masks = torch.zeros((0, *image.size[::-1]), dtype=torch.float32, device=DEVICE)

    label_conf_maps = {label: [] for label in labels_list}
    for detected_label, mask, confidence in zip(detected_labels, masks, confidences):
        if detected_label in label_conf_maps:
            label_conf_maps[detected_label].append(mask.mul_(confidence))  # In-place multiplication

    # Compute pixel-wise maximum across all masks per label
    class_masks = {
        label: torch.amax(torch.stack(label_conf_maps[label]), dim=0) if label_conf_maps[label]
        else torch.zeros(image.size[::-1], dtype=torch.float32, device=DEVICE)
        for label in labels_list
    }

    # Move results back to CPU for further processing
    class_masks = {label: class_masks[label].cpu().numpy() for label in class_masks}

    # Explicit memory cleanup
    del inputs, outputs, results, input_boxes, masks, label_conf_maps, confidences
    torch.cuda.synchronize()  # Ensure all GPU ops finish before clearing memory
    torch.cuda.empty_cache()

    return class_masks, detected_labels, concatenated_labels

warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

print(tf.test.is_gpu_available())

tf.get_logger().setLevel("WARN")  # Suppress warnings in logs

os.makedirs("cache", exist_ok=True)
device = "cuda" if tf.config.list_physical_devices("GPU") else "cpu"

def clean():
    import gc

    from tensorflow.keras.backend import clear_session

    # Clear session
    clear_session()
    gc.collect()

def hash_string(input_string):
    """Create a hash for a given input string."""
    return hashlib.md5(input_string.encode()).hexdigest()

def get_temp_file_names(video_path, output_dir="temp", extra_classes=[]):
    base_dir = os.path.dirname(video_path)
    output_dir = os.path.join(base_dir,  output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    prefix = os.path.basename(video_path)[:4]  # Ensure prefix is valid
    prefix = os.path.basename(video_path)
    video_hash = hashlib.md5(prefix.encode()).hexdigest()
  
    l_path = os.path.join(output_dir, f"{prefix}_gray_{video_hash}.mp4")
    deoldify_a_path = os.path.join(output_dir, f"{prefix}_deoldify_a_{video_hash}.mp4")
    deoldify_b_path = os.path.join(output_dir, f"{prefix}_deoldify_b_{video_hash}.mp4")
    deoldify_path = os.path.join(output_dir, f"{prefix}_deoldify_{video_hash}.mp4")

    class_mask_paths = {
        cls: os.path.join(output_dir, f"{prefix}_{cls}_{video_hash}.mp4")
        for cls in extra_classes
    }

    print(f"Output Directory: {output_dir}")
    print(f"Gray Path: {l_path}")
    print(f"deoldify_a Path: {deoldify_a_path}")
    print(f"deoldify_b Path: {deoldify_b_path}")
    print(f"deoldify Path: {deoldify_path}")

    print(f"Class Mask Paths: {class_mask_paths}")

    return  l_path,  deoldify_a_path, deoldify_b_path, class_mask_paths, deoldify_path

def denoise3d_video(input_video_path, output_dir="cache"):
    """
    Apply denoise3d filter to a video using ffmpeg and return the denoised file path.

    Args:
        input_video_path (str): Path to the input video file.
        output_dir (str): Directory to save the denoised video file.

    Returns:
        str: Path to the denoised video file.
    """
    # Ensure the input file exists
    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"Input video file not found: {input_video_path}")

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate the output file path
    input_filename = os.path.basename(input_video_path)
    output_filename = os.path.join(output_dir, f"{input_filename}.denoised3d.mp4")
    # Construct the ffmpeg command
    ffmpeg_command = [
        "ffmpeg",
        "-i",
        input_video_path,  # Input file
        "-vf",
        "hqdn3d",  # Apply denoise3d filter
        "-c:v",
        "libx264",  # Use H.264 codec for output
        "-preset",
        "fast",  # Use fast preset for better perf
        "-crf",
        "18",  # Set constant rate factor for quality
        "-an",  # Drop audio
        "-y",  # Overwrite output file if it exists
        output_filename,
    ]

    try:
        import shutil

        # Run the ffmpeg command
        subprocess.run(
            ffmpeg_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        shutil.move(output_filename, input_video_path)
        print(f"Denoised video saved to: {output_filename}")
        return input_video_path
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while denoising video: {e.stderr.decode()}")
        raise e


# Preprocessing function to generate depth, L-channel, contour, and additional masks
def preprocess_video_if_needed(
    video_path,  l_path, deoldify_a_path, deoldify_b_path, class_mask_paths, deoldify_path,
    sam2_predictor, processor, grounding_model, TEXT_PROMPT, DEVICE, labels_list, frame_size=(256, 256)
):
    """
    Preprocesses a video to extract depth, L-channel, contour, and additional class masks.

    Args:
        video_path (str): Path to input video.
        depth_path (str): Path to save depth map video.
        l_path (str): Path to save L-channel video.
        contour_path (str): Path to save contour video.
        class_mask_paths (dict): Dictionary mapping class names to their video paths.
        sam2_predictor, processor, grounding_model, TEXT_PROMPT, DEVICE: Models and parameters for segmentation.
        frame_size (tuple): Target frame size.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_paths = [l_path, deoldify_a_path, deoldify_b_path] + list(class_mask_paths.values()) + [deoldify_path]

    if all(os.path.exists(path) for path in all_paths):
        print(f"Preprocessed files already exist: {all_paths}")
        return

    print(f"Processing video: {video_path}")

    # Initialize video capture and writers
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    exception = False
    # Create writers
    l_writer = cv2.VideoWriter(l_path, fourcc, fps, frame_size, isColor=False)
    deoldify_a_writer = cv2.VideoWriter(deoldify_a_path, fourcc, fps, frame_size, isColor=False)
    deoldify_b_writer = cv2.VideoWriter(deoldify_b_path, fourcc, fps, frame_size, isColor=False)
    deoldify_writer = cv2.VideoWriter(deoldify_path, fourcc, fps, frame_size, isColor=True)
    class_writers = {cls: cv2.VideoWriter(path, fourcc, fps, frame_size, isColor=False) for cls, path in class_mask_paths.items()}

    try:
        with torch.no_grad():
            with tqdm(total=total_frames, desc="Processing frames") as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    sam_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_rgb = cv2.resize(frame_rgb, frame_size)

                    # L-channel
                    lab_frame = rgb2lab(frame_rgb.astype(np.float32) / 255.0)
                    sam_lab_frame = rgb2lab(sam_frame_rgb.astype(np.float32) / 255.0)
                    l_channel = (lab_frame[:, :, 0] / 100.0 * 255).astype(np.uint8)
                    sam_l_channel = (sam_lab_frame[:, :, 0] / 100.0 * 255).astype(np.uint8)
                    l_writer.write(l_channel)

                    PIL_frame_rgb = Image.fromarray(l_channel)
                    deoldify_frame = deoldify_inference(PIL_frame_rgb)
                    deoldify_lab_frame = rgb2lab(deoldify_frame.astype(np.float32) / 255.0)
                   # deoldify_a_channel = (deoldify_lab_frame[:, :, 1] / 100.0 * 255).astype(np.uint8)
                    deoldify_a_channel = (deoldify_lab_frame[:, :, 1] + 128).astype(np.uint8)
                    deoldify_a_writer.write(deoldify_a_channel)
                   # deoldify_b_channel = (deoldify_lab_frame[:, :, 2] / 100.0 * 255).astype(np.uint8)
                    deoldify_b_channel = (deoldify_lab_frame[:, :, 2] + 128).astype(np.uint8)
                    deoldify_b_writer.write(deoldify_b_channel)
                    deoldify_frame = cv2.cvtColor(deoldify_frame, cv2.COLOR_RGB2BGR)
                    deoldify_writer.write(deoldify_frame)

                    # Class masks using SAM2 + Grounding DINO
                    image_pil = Image.fromarray(sam_l_channel)
                    MAX_DIM = 1024
                    w, h = image_pil.size
                    if max(w, h) > MAX_DIM:
                        scale = MAX_DIM / max(w, h)
                        new_size = (int(w * scale), int(h * scale))
                        image_pil = image_pil.resize(new_size, Image.Resampling.LANCZOS)
                    class_masks,_,_ = extract_class_masks(image_pil, sam2_predictor, processor, grounding_model, TEXT_PROMPT, DEVICE, labels_list)

                    for class_name, mask in class_masks.items():
                       # print("preprocess_video_if_needed",class_masks.keys(),class_name)
                        mask_resized = cv2.resize((mask * 255).astype(np.uint8), frame_size)
                        class_writers[class_name].write(mask_resized)

                    pbar.update(1)

    except Exception as e:
        print(f"Error processing video: {e}")
        traceback.print_exc()
        exception = True
    finally:
        cap.release()      
        l_writer.release()
        deoldify_a_writer.release()
        deoldify_b_writer.release()
        deoldify_writer.release()
        for writer in class_writers.values():
            writer.release()
        torch.cuda.empty_cache()

        if not exception:
            for path in all_paths:
                denoise3d_video(path)
        else:
            for path in all_paths:
                os.remove(path)

# VideoDataLoader with additional class masks
class VideoDataLoader(Sequence):
    def __init__(self, video_path, batch_size, frame_size=(256, 256), noise_std=0.05, training_mode=False, extra_classes=[]):
        """
        VideoDataLoader with dynamic masks for detected classes.

        Args:
            video_path (str): Input video path.
            batch_size (int): Frames per batch.
            frame_size (tuple): Target frame size.
            noise_std (float): Gaussian noise for augmentation.
            training_mode (bool): Whether to shuffle batches.
            extra_classes (list): List of detected classes to include as masks.
        """
        self.video_path = video_path
        self.batch_size = batch_size
        self.frame_size = frame_size
        self.noise_std = noise_std
        self.training_mode = training_mode

        self.l_path,  self.deoldify_a_path, self.deoldify_b_path, self.class_mask_paths, self.deoldify_path = get_temp_file_names(video_path, "cache", extra_classes)

        preprocess_video_if_needed(self.video_path,  self.l_path, self.deoldify_a_path, self.deoldify_b_path, self.class_mask_paths, self.deoldify_path, sam2_predictor, processor, grounding_model, TEXT_PROMPT, "cuda", labels, self.frame_size)

        self.cap = cv2.VideoCapture(self.video_path)
        self.l_cap = cv2.VideoCapture(self.l_path)
        self.deoldify_a_cap = cv2.VideoCapture(self.deoldify_a_path)
        self.deoldify_b_cap = cv2.VideoCapture(self.deoldify_b_path)
        self.class_caps = {cls: cv2.VideoCapture(path) for cls, path in self.class_mask_paths.items()}

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.batch_indices = []
        self._compute_batches()

    def _compute_batches(self):
        """Precompute batch indices."""
        self.batch_indices = [(start, min(start + self.batch_size, self.total_frames))
                              for start in range(0, self.total_frames, self.batch_size)]
        if self.training_mode:
            random.shuffle(self.batch_indices)

    def __len__(self):
        return len(self.batch_indices)

    def __getitem__(self, idx):
        """
        Fetch a batch of frames along with additional class-based masks.

        Args:
            idx: Batch index.

        Returns:
            Tuple of stacked inputs (L, Depth, Contour, Extra Classes), AB channels, and original RGB frames.
        """
        if idx >= len(self.batch_indices):
            raise IndexError(f"Index {idx} out of range for batch_indices of size {len(self.batch_indices)}")

        start_frame, end_frame = self.batch_indices[idx]

        # Set the frame position in all video captures
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        self.l_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        self.deoldify_a_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        self.deoldify_b_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for cap in self.class_caps.values():
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Initialize batch storage
        frames, L_batch, deoldify_a_batch, deoldify_b_batch, AB_batch =  [], [], [], [], []
        Class_Batches = {cls: [] for cls in self.class_caps.keys()}

        for _ in range(start_frame, end_frame):
            ret, bgr_frame = self.cap.read()
            ret_l, gray_frame = self.l_cap.read()
            ret_deoldify_a, deoldify_a_frame = self.deoldify_a_cap.read()
            ret_deoldify_b, deoldify_b_frame = self.deoldify_b_cap.read()
            class_frames = {cls: cap.read()[1] for cls, cap in self.class_caps.items()}

            # Process RGB frame
            frame_rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, self.frame_size)
            frames.append(frame_resized.astype(np.float32) / 255.0)

            # Convert frame to LAB
            lab_frame = rgb2lab(frame_resized.astype(np.float32) / 255.0)
            L = lab_frame[:, :, 0] / 100.0  # Normalize to [0, 1]

            L_batch.append(L[..., np.newaxis])

            # Process depth, contour, and additional class masks
            deoldify_a_batch.append((cv2.cvtColor(deoldify_a_frame, cv2.COLOR_BGR2GRAY) / 255.0)[..., np.newaxis])
            deoldify_b_batch.append((cv2.cvtColor(deoldify_b_frame, cv2.COLOR_BGR2GRAY) / 255.0)[..., np.newaxis])
            for cls, cls_frame in class_frames.items():
                Class_Batches[cls].append((cv2.cvtColor(cls_frame, cv2.COLOR_BGR2GRAY) / 255.0)[..., np.newaxis])

            # Process AB Channels
            AB = np.clip((lab_frame[:, :, 1:] / 128.0 + 1.0) / 2.0, 0, 1)
            AB_batch.append(AB)

        # Convert to NumPy arrays
        L_batch = np.array(L_batch)
        deoldify_a_batch = np.array(deoldify_a_batch)
        deoldify_b_batch = np.array(deoldify_b_batch)
        AB_batch = np.array(AB_batch)
        RGB_batch = np.array(frames)
        Class_Batches = {cls: np.array(batch) for cls, batch in Class_Batches.items()}

        # Ensure proper dimensions
        assert L_batch.ndim == 4, "Input batches must be 4D"

        # Concatenate all inputs including extra classes
        stacked_inputs = np.concatenate([L_batch, deoldify_a_batch, deoldify_b_batch] + list(Class_Batches.values()), axis=-1)
        #print("new************************************************")
        return (
            stacked_inputs.astype(np.float32),  # Shape: (batch_size, height, width, num_channels)
            AB_batch.astype(np.float32),  # Shape: (batch_size, height, width, 2)
            RGB_batch.astype(np.float32),  # Original frames
        )

    def __del__(self):
        """Release video capture handles when done."""
        if self.cap.isOpened():
            self.cap.release()
        if self.l_cap.isOpened():
            self.l_cap.release()
        if self.deoldify_a_cap.isOpened():
            self.deoldify_a_cap.release()
        if self.deoldify_b_cap.isOpened():
            self.deoldify_b_cap.release()
        for cap in self.class_caps.values():
            if cap.isOpened():
                cap.release()

class VideoScenePrewriter:
    def __init__(self, input_files, skip_size, frame_size=(256, 256)):
        """
        Initialize the SceneDataLoader.

        Args:
            input_files: List of video files to read from...
            batch_size: Number of frames per batch.
            frame_size: Tuple (width, height) for resizing frames.
        """
        self.skip_size = skip_size
        self.frame_size = frame_size
        self.videos = input_files
        prefix = os.path.basename(input_files[0])[:4]  # Ensure prefix is valid
        video_hash = hashlib.md5(prefix.encode()).hexdigest()
        self.video_path = f"{prefix}_scenes.mp4"


    def load_videos(self):
        if not os.path.exists(self.video_path):
            # Initialize video capture and writers
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            (w, h) = self.frame_size
            scene_writer = cv2.VideoWriter(self.video_path, fourcc, 30, self.frame_size)
            print(f"Writing the compiled file to {self.video_path}")
            try:
                for video_file in self.videos:
                    print(f"Processing {video_file}...")
                    idx = 0
                    cap = cv2.VideoCapture(video_file)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    try:
                        with tqdm(
                            total=total_frames,
                            desc=f"Reading Frames from {video_file}",
                        ) as pbar:
                            while idx < total_frames:
                                ret, bgr_frame = cap.read()
                                idx += 1
                                pbar.update(1)
                                if not ret:
                                    continue
                                # Convert frame to smaller size
                                if idx % self.skip_size == 0:
                                    frame_rgb = cv2.resize(
                                        bgr_frame,
                                        self.frame_size,
                                    )
                                    scene_writer.write(frame_rgb)
                    finally:
                        if cap and cap.isOpened():
                            cap.release()
            except Exception as e:
                print(f"Exception occurred writing the scene file {e}")
                traceback.print_exc()
            finally:
                scene_writer.release()
        return self.video_path



import os

import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from skimage.color import lab2rgb
from tensorflow.keras.callbacks import Callback

def generate_rgb(L, AB):
    """
    Convert L and AB channels to RGB using LAB-to-RGB conversion.

    Args:
        L (np.ndarray): L channel in range [0, 1].
        AB (np.ndarray): AB channels in range [-1, 1] or normalized.

    Returns:
        np.ndarray: RGB image in uint8 format ([0, 255]).
    """
    # Ensure inputs are numpy arrays
    L_numpy = np.clip(L.astype(np.float32), 0, 1)  # Normalize L if needed
    AB_numpy = AB.astype(np.float32)  # Assuming AB is already normalized appropriately

    # Prepare LAB numpy array
    LAB_numpy = np.zeros((L_numpy.shape[0], L_numpy.shape[1], 3), dtype=np.float32)
    LAB_numpy[:, :, 0] = L_numpy * 100.0  # L channel denormalized to [0, 100]
    LAB_numpy[:, :, 1:] = (AB_numpy * 2.0 - 1.0) * 127.0  # AB scaled to [-128, 127]

    # Convert LAB to RGB using skimage
    rgb_frame = lab2rgb(LAB_numpy)  # LAB-to-RGB conversion, output in [0, 1]
    rgb_frame_uint8 = np.clip((rgb_frame * 255.0), 0, 255).astype(np.uint8)  # [0, 255]

    return rgb_frame_uint8


class VisualizeCallback(Callback):
    def __init__(
        self, generator, validation_data, output_dir="visualizations", interval=1
    ):
        """
        Callback to visualize predictions during training.

        Args:
            generator: The generator model.
            validation_data: The validation data loader.
            output_dir: Directory to save visualizations.
            interval: Interval (in epochs) at which to generate visualizations.
        """
        super().__init__()
        self.generator = generator
        self.validation_data = validation_data
        self.output_dir = output_dir
        self.interval = interval
        os.makedirs(output_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.interval == 0:
            print(f"Visualizing predictions for epoch {epoch + 1}...")
            if self.validation_data:
                random.shuffle(self.validation_data.batch_indices)
            stacked_inputs, real_AB, real_rgbs = next(
                iter(self.validation_data)
            )  # Fetch a batch of data

            # Predict AB channels from stacked_inputs
            predicted_AB = self.generator.predict(stacked_inputs)

            for i in range(
                min(4, stacked_inputs.shape[0])
            ):  # Iterate through frames in the batch
                # Extract L channel from stacked_inputs
                L_channel = stacked_inputs[
                    i, :, :, 0
                ]  # Assuming L is the first channel

                # Extract predicted AB and denormalize it back to [-128, 127]
                pred_AB = predicted_AB[i, :, :, :2]

                # Reconstruct predicted RGB from L and predicted AB
                reconstructed_rgb = generate_rgb(L_channel, pred_AB)

                # Convert real RGB to uint8 for visualization
                real_rgb = (real_rgbs[i] * 255).astype(np.uint8)

                # Plot and save results
                plt.figure(figsize=(15, 5))

                # Predicted AB reconstruction
                plt.subplot(1, 3, 1)
                plt.title("Reconstructed LAB")
                plt.imshow(reconstructed_rgb.astype(np.uint8))
                plt.axis("off")

                # Ground truth RGB
                plt.subplot(1, 3, 2)
                plt.title("Ground Truth RGB")
                plt.imshow(real_rgb)
                plt.axis("off")

                # L Channel (grayscale)
                plt.subplot(1, 3, 3)
                plt.title("L Channel")
                plt.imshow((L_channel * 255).astype(np.uint8), cmap="gray")
                plt.axis("off")

                # Save the figure
                save_path = os.path.join(
                    self.output_dir, f"epoch_{epoch + 1}_frame_{i + 1}.png"
                )
                plt.savefig(save_path, bbox_inches="tight")
                plt.close()


from skimage.color import rgb2lab

##-need to check
def dev_loss(y_true, y_pred):
    pred_AB = y_pred[..., :2]  # First two channels are AB
    mse_pred_AB = tf.reduce_mean(tf.abs(y_true - pred_AB))
    return mse_pred_AB

import cv2
import random
import numpy as np

def split_video_randomly(input_video_path, output_video_1, output_video_2, ratio=0.95):
    cap = cv2.VideoCapture(input_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define codec and create VideoWriter objects
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' codec for MP4 format
    out1 = cv2.VideoWriter(output_video_1, fourcc, fps, (frame_width, frame_height))
    out2 = cv2.VideoWriter(output_video_2, fourcc, fps, (frame_width, frame_height))

    # Randomly select frames for validation (5%)
    validation_indices = set(random.sample(range(total_frames), int(total_frames * (1 - ratio))))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in validation_indices:
            out2.write(frame)  # 5% frames go into second video
        else:
            out1.write(frame)  # 95% frames go into first video

        frame_idx += 1

    cap.release()
    out1.release()
    out2.release()
    print(f"Video split completed: {output_video_1} (95%) and {output_video_2} (5%)")

from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def train(input_files, val_files, epochs=5, output_dir="output"):
    """
    Train the generator model with validation and automatic checkpointing.
    """
    try:
        import shutil

        os.makedirs(output_dir, exist_ok=True)
        weights_path = os.path.join(output_dir, "generator_weights.weights.h5")

        # Optimizer
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-4, decay_steps=1000, decay_rate=0.99
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        # Build base generator model
        print("Creating generator...")
        base_generator = ColoringModel()
        print(base_generator.input_shape, base_generator.output_shape)

        print("No weights loaded.")

        # Custom multi-loss model wrapper
        print("Wrapping with custom multi-loss model...")
        class MyModel(tf.keras.Model):
            def __init__(self, base_model, alpha=0, beta=1, gamma=1):
                super(MyModel, self).__init__()
                self.model = base_model
                self.alpha = alpha
                self.beta = beta
                self.gamma = gamma

            def compile(self, optimizer, **kwargs):
                super().compile(**kwargs)
                self.optimizer = optimizer

            def train_step(self, data):
                x, y, _ = data  # x: stacked_inputs, y: AB GT, _: RGB frame (not used)

                with tf.GradientTape() as tape:
                    y_pred = self.model(x, training=True)

                    loss1 = dev_loss(y, y_pred)
                   # loss2 = perceptual_loss(y, y_pred)/1000000
                    loss2 = perceptual_loss(y, y_pred)
                    loss3 = third_loss_from_input(x, y_pred)  # deoldify_a & b from x[..., 3:5]

                    total_loss = self.alpha * loss1 + self.beta * loss2 + self.gamma * loss3

                grads = tape.gradient(total_loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                return {
                    "loss": total_loss,
                    "dev_loss": loss1,
                    "perceptual_loss": loss2,
                    "deoldify_loss": loss3,
                }

            def test_step(self, data):
                x, y, _ = data

                y_pred = self.model(x, training=False)

                loss1 = dev_loss(y, y_pred)
                #loss2 = perceptual_loss(y, y_pred) / 1000000  # normalize if needed
                loss2 = perceptual_loss(y, y_pred)
                loss3 = third_loss_from_input(x, y_pred)

                total_loss = self.alpha * loss1 + self.beta * loss2 + self.gamma * loss3

                return {
                    "loss": total_loss,
                    "dev_loss": loss1,
                    "perceptual_loss": loss2,
                    "deoldify_loss": loss3,
                }

            def call(self, inputs, training=False):
                return self.model(inputs, training=training)

            def compute_loss(self, x, y, y_pred, sample_weight=None):
                # We compute loss ourselves in train_step, so just return 0
                return 0.0

        # Instantiate and compile model
        generator = MyModel(base_generator, alpha=0, beta=1, gamma=1)
        generator.compile(optimizer=optimizer)
        print("Compiled custom generator.")

        # Training and validation data loaders
        print("Loading training video file...")
        trainloader = VideoDataLoader(
            VideoScenePrewriter(input_files, 1, (256, 256)).load_videos(),
            batch_size=8,
            frame_size=(256, 256),
            training_mode=True,
            extra_classes=labels
        )

        print("Loading validation video file...")
        valloader = VideoDataLoader(
            VideoScenePrewriter(val_files, 1, (256, 256)).load_videos(),
            batch_size=8,
            frame_size=(256, 256),
            training_mode=False,
            extra_classes=labels
        )

        # Callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(output_dir, "best_weights_epoch_{epoch:04d}.weights.h5"),
                save_weights_only=True,
                save_best_only=True,
                monitor="val_loss",
                verbose=1,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(output_dir, "weights_epoch_{epoch:04d}.weights.h5"),
                save_weights_only=True,
                save_best_only=False,
                monitor="val_loss",
                verbose=1,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=100, restore_best_weights=True, verbose=1
            ),
            VisualizeCallback(
                generator,
                valloader,
                output_dir=os.path.join(output_dir, "visualizations"),
                interval=1,
            ),
        ]

        # Training
        print("Starting training...")
        generator.build(input_shape=(None, 256, 256, 12))
        # generator.load_weights("epochs_50_3_loss_functions/generator_weights.weights.h5.final.weights.h5")
        # print("loaded epochs_50_3_loss_functions/generator_weights.weights.h5.final.weights.h5")
        print("no weights")
        history = generator.fit(
            trainloader,
            validation_data=valloader,
            epochs=100,
            callbacks=callbacks,
            verbose=1,
        )

        # Save final weights
        print("Saving final weights...")
        generator.save_weights(weights_path + ".final.weights.h5")

    except Exception as e:
        print("An error occurred during training:")
        traceback.print_exc()
    finally:
        if "trainloader" in vars():
            del trainloader
        if "valloader" in vars():
            del valloader
        if "generator" in vars():
            del generator
        clean()

import logging
from tqdm import tqdm
import sys

# Reset logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.WARNING)  # Reduce log verbosity

# Clear tqdm
tqdm._instances.clear()

def load_generator_for_inference(weights_path):
    """
    Rebuilds the generator model and loads the saved weights.

    Args:
        weights_path: Path to the saved weights file.
        input_shape: Input shape for the generator model.

    Returns:
        Generator model with loaded weights.
    """
    generator = ColoringModel()
    #generator.load_weights(weights_path, by_name=True, skip_mismatch=True)
    generator.load_weights(weights_path, skip_mismatch=True)
    return generator

# At top-level of the file
class MyModel(tf.keras.Model):
    def __init__(self, base_model, alpha=0.3, beta=0.7, gamma=0.5):
        super(MyModel, self).__init__()
        self.model = base_model
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)

    # Optional: Keep train_step/test_step if needed for further training

def load_generator_for_inference(weights_path):
    base_model = ColoringModel()
    model = MyModel(base_model, alpha=0.3, beta=0.7, gamma=0.5)

    # Force model to build by running dummy input through it
    dummy_input = tf.random.uniform((1, 256, 256, 12))  # NUM_CHANNELS = e.g., 8
    model(dummy_input, training=False)

    # Load weights into the wrapped model
    model.load_weights(weights_path)

    return model

def recolor_video_with_dataloader(
    generator, video_path, output_path,  resize, batch_size=32,  frame_size=(256, 256),
):
    """
    Recolor a grayscale video using the trained generator model and VideoDataLoader.

    Args:
        generator: Trained U-Net generator model.
        video_path: Path to the input grayscale video.
        output_path: Path to save the recolored video.
        batch_size: Number of frames to process in a batch.
        frame_size: Size of frames to process (H, W).
    """
    from IPython.display import display
    from PIL import Image as PILImage
    from skimage.color import lab2rgb

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing video: {video_path}")
    print(f"Total frames: {total_frames}, FPS: {fps}, Output size: {frame_size}")
    cap.release()
    # Initialize batch processing
    batch_frames = []
    processed_frames = []

    try:
        loader = VideoDataLoader(
            video_path,
            batch_size=batch_size,
            frame_size=frame_size,
            training_mode=False,
            extra_classes = labels
        )
        torch.cuda.synchronize()  # Ensure all GPU ops finish before clearing memory
        torch.cuda.empty_cache()
        generator = load_generator_for_inference(generator)
        with tqdm(total=total_frames, desc="Processing Frames") as pbar:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            frame_width, frame_height = resize
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            for stacked_inputs, AB, rgb in loader:
                # Predict AB channels using the generator
                predicted_ab_batch = generator.predict(stacked_inputs, verbose=0)
                for i in range(len(predicted_ab_batch)):
                    # Extract L channel and predicted AB channels
                    L_numpy = stacked_inputs[i, ..., 0]  # Shape: (256, 256)
                    # AB_numpy = np.clip(
                    #     predicted_ab_batch[i, ...], 0, 1
                    # )  # Clip AB to [-1, 1]
                    AB_numpy = predicted_ab_batch[i, :, :, :2]
                    rgb_frame = generate_rgb(L_numpy, AB_numpy)
                    pbar.update(1)

                    # Write frame to output video
                    out.write(
                        cv2.resize(
                            cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR),
                             (frame_width, frame_height),
                            interpolation=cv2.INTER_CUBIC
                        )
                    )
            # Release resources
            out.release()
            print(f"Recolored video saved to: {output_path}")

    except Exception as e:
        print(f"Error while processing video: {e}")

    # finally:
    #     clean()
    finally:
        # 🧹 Clean up TensorFlow and Torch memory
        try:
            tf.keras.backend.clear_session()
            del generator
            gc.collect()
        except Exception as del_err:
            print(f"Could not delete generator: {del_err}")
                
        torch.cuda.empty_cache()
        clean()

def recolor_video_with_external_luminance(
    generator,
    video_path,        # AB prediction input
    l_video_path,      # Source video for L channel
    output_path,       # Final RGB output video
    frame_size=(256, 256),
    batch_size=8,
):
    from IPython.display import display
    from PIL import Image as PILImage
    from skimage.color import lab2rgb

    # Open the L-channel video
    El_cap = cv2.VideoCapture(l_video_path)
    fps = int(El_cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(El_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(El_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(El_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Using L from: {l_video_path}")
    print(f"Predicting AB from: {video_path}")
    print(f"Output → {output_path} at resolution {(frame_width, frame_height)}")
    print(f"Processing video: {video_path}")
    print(f"Total frames: {total_frames}, FPS: {fps}, Output size: {frame_size}")
    
    # Output writer
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )

    try:
        loader = VideoDataLoader(
            video_path,
            batch_size=batch_size,
            frame_size=frame_size,
            training_mode=False,
            extra_classes=labels  # if needed
        )
        torch.cuda.synchronize()  # Ensure all GPU ops finish before clearing memory
        torch.cuda.empty_cache()
        generator = load_generator_for_inference(generator)
        frame_idx = 0

        for stacked_inputs, _, _ in tqdm(loader, desc="Recoloring frames"):
            predicted_ab_batch = generator.predict(stacked_inputs, verbose=0)

            for i in range(len(predicted_ab_batch)):
                ret, l_frame = El_cap.read()
                if not ret:
                    print(f"[WARN] Ran out of L frames at index {frame_idx}")
                    break

                # Convert L frame to grayscale in [0,1]
                l_gray = cv2.cvtColor(l_frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

                # Resize AB prediction to match L resolution
                ab_pred = predicted_ab_batch[i, :, :, :2]
                ab_resized = cv2.resize(ab_pred, (l_gray.shape[1], l_gray.shape[0]), interpolation=cv2.INTER_CUBIC)

                # Generate final RGB using your custom function
                rgb_frame = generate_rgb(l_gray, ab_resized)

                # Write output
                out.write(cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
                frame_idx += 1

    # finally:
    #     clean()
    finally:
        # 🧹 Clean up TensorFlow and Torch memory
        try:
            El_cap.release()
            out.release()
            tf.keras.backend.clear_session()
            del generator
            gc.collect()
        except Exception as del_err:
            print(f"Could not delete generator: {del_err}")
               
        torch.cuda.empty_cache()
        clean()

print("cell 1 completed")

def run_colorization_cached(input_path: str, generator_path: str, output_video_path : str) -> str:
    """
    Run UNet-based colorization on the given grayscale video (input_path).
    Saves the colorized output as 'color.mp4' in the same directory.
    
    Parameters:
        input_path: Path to grayscale video (e.g., *_bw.mp4).
        generator_path: Path to UNet weights/model file.
    
    Returns:
        Full path to the colorized video (color.mp4).
    """   
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"[ERROR] Grayscale video not found at: {input_path}")

    if os.path.exists(output_video_path):
        print(f"[CACHE] Colorized output already exists: {output_video_path}")
        return str(output_video_path)

    print("[INFO] Running UNET colorisation...")
    from Utils.unet_utils import load_generator_for_inference, recolor_video_with_external_luminance
    try:
        recolor_video_with_external_luminance(
            generator_path,
            str(input_path),
            str(input_path),
            str(output_video_path),
            frame_size=(256, 256)
        )
    except (Exception, KeyboardInterrupt) as e:
        print(f"[ERROR] UNET Colorisation interrupted: {e}")
        if os.path.exists(output_video_path):
            os.remove(output_video_path)
        cache_folder = os.path.join(os.path.dirname(input_path), "cache")
        if os.path.exists(cache_folder) and os.path.isdir(cache_folder):
            print(f"[CLEANUP] Removing cache folder: {cache_folder}")
            shutil.rmtree(cache_folder)
        raise

    from Utils.main_utils import repair_video_file
    repair_video_file(str(output_video_path))
    print(f"[DONE] Colorization completed with external Luminance→ {output_video_path}")
    return str(output_video_path)



def run_unet_colorization_cached_subprocess(input_bw_video, unet_weights, output_video_path):
    """
    Runs UNet-based colorization via subprocess with caching logic.
    This wraps a subprocess call to `run_colorization_cached()` in Utils.main_utils.

    Parameters:
        input_bw_video: path to grayscale video (e.g., *_bw.mp4)
        unet_weights: path to UNet weights
        first_path: root/original video path (used for cache folder naming)

    Returns:
        Path to the final colorized video
    """
    # Free up GPU memory
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()

    print("[INFO] Launching UNET colorization subprocess...")
    from Utils.unet_utils import run_colorization_cached
    try:
        subprocess.run([
            sys.executable, "-c",
            (
                "from Utils.unet_utils import run_colorization_cached; "
                f"run_colorization_cached('{input_bw_video}', '{unet_weights}', '{output_video_path}')"
            )
        ], check=True)
        return output_video_path

    except (subprocess.CalledProcessError, Exception, KeyboardInterrupt) as e:
        print(f"[ERROR] UNET Colorisation subprocess failed: {e}")

        # Remove output video if partially created
        if os.path.exists(output_video_path):
            print(f"[CLEANUP] Removing partial output video: {output_video_path}")
            os.remove(output_video_path)

        # Remove related cache folder
        cache_folder = os.path.join(os.path.dirname(input_bw_video), "cache")
        if os.path.exists(cache_folder) and os.path.isdir(cache_folder):
            print(f"[CLEANUP] Removing cache folder: {cache_folder}")
            shutil.rmtree(cache_folder)
        raise

import os
import sys
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run UNet-based colorization on a B&W video (with subprocess isolation).")
    parser.add_argument("input_bw_video", help="Path to the grayscale (B&W) input video")
    parser.add_argument("unet_weights", help="Path to the trained UNet weights (.h5)")
    parser.add_argument("output_path", help="Output path used for cache structuring")

    args = parser.parse_args()

    if not os.path.exists(args.input_bw_video):
        print(f"[ERROR] Input B&W video does not exist: {args.input_bw_video}")
        sys.exit(1)

    if not os.path.exists(args.unet_weights):
        print(f"[ERROR] UNet weights not found: {args.unet_weights}")
        sys.exit(1)

    if os.path.exists(args.output_path):
        print(f"[CACHE] Colorized output already exists: {args.output_path}")
        sys.exit(1)

    try:
        final_colorized_video = run_unet_colorization_cached_subprocess(
            input_bw_video=args.input_bw_video,
            unet_weights=args.unet_weights,
            output_video_path=args.output_path
        )
        print(f"✔️ Colorized video saved at: {final_colorized_video}")

    except Exception as e:
        print(f"[ERROR] Colorization failed: {e}")
        sys.exit(1)
