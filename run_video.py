# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import argparse
import sys
import shutil
import subprocess
import cv2
from pathlib import Path
from tqdm import tqdm


def run_cmd(command):
    """Run shell command with keyboard interrupt handling."""
    try:
        subprocess.call(command, shell=True)
    except KeyboardInterrupt:
        print("Process interrupted")
        sys.exit(1)


def extract_frames(video_path, frames_dir):
    """Extract frames from a video file."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Extracting {frame_count} frames at {fps:.2f} FPS...")
    for i in tqdm(range(frame_count), desc="Extracting Frames", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = frames_dir / f"frame_{i:06d}.png"
        cv2.imwrite(str(frame_path), frame)
    cap.release()
    print(f"[INFO] Frame extraction completed: {len(list(frames_dir.glob('*.png')))} frames")
    return fps


def assemble_video(frames_dir, output_path, fps):
    """Reassemble frames into a video."""
    print(f"[INFO] Assembling video from restored frames → {output_path}")
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%06d.png"),
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(output_path)
    ]
    subprocess.call(cmd)
    print("[INFO] Video assembly completed.")


def run_restoration_pipeline(input_folder, output_folder, gpu="0", checkpoint="Setting_9_epoch_100", with_scratch=True, hr=True):
    """Run the integrated Bringing-Old-Photos-Back-to-Life 4-stage restoration."""
    os.makedirs(output_folder, exist_ok=True)
    gpu1 = gpu

    input_folder = os.path.abspath(input_folder)
    output_folder = os.path.abspath(output_folder)

    main_environment = os.getcwd()

    # Stage 1: Global Restoration
    print("\n[Stage 1] Overall Restoration")
    os.chdir("/opt/bringing_old_photos_back_to_life/Global")
    stage_1_output_dir = os.path.join(output_folder, "stage_1_restore_output")
    os.makedirs(stage_1_output_dir, exist_ok=True)

    if not with_scratch:
        cmd = (
            f"python test.py --test_mode Full --Quality_restore "
            f"--test_input {input_folder} "
            f"--outputs_dir {stage_1_output_dir} "
            f"--gpu_ids {gpu1}"
        )
        run_cmd(cmd)
    else:
        mask_dir = os.path.join(stage_1_output_dir, "masks")
        new_input = os.path.join(mask_dir, "input")
        new_mask = os.path.join(mask_dir, "mask")
        os.makedirs(mask_dir, exist_ok=True)

        cmd1 = (
            f"python detection.py --test_path {input_folder} "
            f"--output_dir {mask_dir} --input_size full_size --GPU {gpu1}"
        )
        run_cmd(cmd1)

        hr_suffix = " --HR" if hr else ""
        cmd2 = (
            f"python test.py --Scratch_and_Quality_restore "
            f"--test_input {new_input} --test_mask {new_mask} "
            f"--outputs_dir {stage_1_output_dir} "
            f"--gpu_ids {gpu1}{hr_suffix}"
        )
        run_cmd(cmd2)

    stage_1_results = os.path.join(stage_1_output_dir, "restored_image")
    stage_4_output_dir = os.path.join(output_folder, "final_output")
    os.makedirs(stage_4_output_dir, exist_ok=True)
    for x in os.listdir(stage_1_results):
        shutil.copy(os.path.join(stage_1_results, x), stage_4_output_dir)

    print("✅ Stage 1 Complete\n")

    # Stage 2: Face Detection
    print("[Stage 2] Face Detection")
    os.chdir("/opt/bringing_old_photos_back_to_life/Face_Detection")
    stage_2_output_dir = os.path.join(output_folder, "stage_2_detection_output")
    os.makedirs(stage_2_output_dir, exist_ok=True)
    if hr:
        cmd = f"python detect_all_dlib_HR.py --url {stage_1_results} --save_url {stage_2_output_dir}"
    else:
        cmd = f"python detect_all_dlib.py --url {stage_1_results} --save_url {stage_2_output_dir}"
    run_cmd(cmd)
    print("✅ Stage 2 Complete\n")

    # Stage 3: Face Enhancement
    print("[Stage 3] Face Enhancement")
    os.chdir("/opt/bringing_old_photos_back_to_life/Face_Enhancement")
    stage_3_output_dir = os.path.join(output_folder, "stage_3_face_output")
    os.makedirs(stage_3_output_dir, exist_ok=True)
    if hr:
        checkpoint = "FaceSR_512"
        cmd = (
            f"python test_face.py --old_face_folder {stage_2_output_dir} "
            f"--old_face_label_folder ./ --tensorboard_log --name {checkpoint} "
            f"--gpu_ids {gpu1} --load_size 512 --label_nc 18 --no_instance "
            f"--preprocess_mode resize --batchSize 1 --results_dir {stage_3_output_dir} --no_parsing_map"
        )
    else:
        cmd = (
            f"python test_face.py --old_face_folder {stage_2_output_dir} "
            f"--old_face_label_folder ./ --tensorboard_log --name {checkpoint} "
            f"--gpu_ids {gpu1} --load_size 256 --label_nc 18 --no_instance "
            f"--preprocess_mode resize --batchSize 4 --results_dir {stage_3_output_dir} --no_parsing_map"
        )
    run_cmd(cmd)
    print("✅ Stage 3 Complete\n")

    # Stage 4: Blending / Warp-back
    print("[Stage 4] Final Face Blending")
    os.chdir("/opt/bringing_old_photos_back_to_life/Face_Detection")
    stage_4_input_image_dir = stage_1_results
    stage_4_input_face_dir = os.path.join(stage_3_output_dir, "each_img")
    os.makedirs(stage_4_output_dir, exist_ok=True)
    if hr:
        cmd = (
            f"python align_warp_back_multiple_dlib_HR.py --origin_url {stage_4_input_image_dir} "
            f"--replace_url {stage_4_input_face_dir} --save_url {stage_4_output_dir}"
        )
    else:
        cmd = (
            f"python align_warp_back_multiple_dlib.py --origin_url {stage_4_input_image_dir} "
            f"--replace_url {stage_4_input_face_dir} --save_url {stage_4_output_dir}"
        )
    run_cmd(cmd)
    print("✅ Stage 4 Complete\n")

    os.chdir(main_environment)
    print("🎉 All restoration stages completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video", type=str, required=True, help="Input old video")
    parser.add_argument("--output_video", type=str, required=True, help="Final restored output video")
    parser.add_argument("--GPU", type=str, default="0", help="GPU IDs to use")
    parser.add_argument("--with_scratch", action="store_true")
    parser.add_argument("--HR", action="store_true")
    parser.add_argument("--checkpoint_name", type=str, default="Setting_9_epoch_100")
    opts = parser.parse_args()

    input_video = Path(opts.input_video).resolve()
    output_video = Path(opts.output_video).resolve()
    work_dir = Path("./temp_video_restore")
    frames_dir = work_dir / "frames"
    restored_dir = work_dir / "restored"
    restored_dir.mkdir(parents=True, exist_ok=True)

    # 1️⃣ Extract frames
    fps = extract_frames(input_video, frames_dir)

    # 2️⃣ Run integrated restoration pipeline
    run_restoration_pipeline(
        input_folder=str(frames_dir),
        output_folder=str(restored_dir),
        gpu=opts.GPU,
        checkpoint=opts.checkpoint_name,
        with_scratch=True,
        hr=True,
    )

    # 3️⃣ Assemble restored frames into video
    final_frames = restored_dir / "final_output"
    if not final_frames.exists():
        raise FileNotFoundError("Final output frames not found after restoration.")
    assemble_video(final_frames, output_video, fps)

    # 4️⃣ Cleanup
    shutil.rmtree(work_dir, ignore_errors=True)
    print(f"[INFO] Done. Final restored video saved at: {output_video}")
