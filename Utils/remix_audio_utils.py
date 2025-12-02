import os
import subprocess
from Utils.main_utils import repair_video_file
def remix_audio(input_path, original_path, output_path):
    """Remix audio from the original video into the DeOldified video."""
    if os.path.exists(output_path):
        print(f"[CACHE] final video found: {output_path}")
        return remixed_file
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-i", original_path,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        output_path,
    ]
    subprocess.run(cmd, capture_output=True)
    repair_video_file(output_path)
    print(f"[INFO] Audio remixed: {output_path}")
    return output_path


def remix_audio_cached(input_path, original_path, output_path):

    if os.path.exists(output_path):
        print(f"[CACHE] Final video found: {output_path}")
        return output_path

    print("[INFO] Running Audio Remix")
    from Utils.remix_audio_utils import remix_audio
    try:
        remix_audio(input_path, original_path, output_path)
    except (Exception, KeyboardInterrupt) as e:
        print(f"[ERROR] Audio Remix interrupted: {e}")
        if os.path.exists(output_path):
            print(f"[CLEANUP] Removing partial output video: {output_path}")
            os.remove(output_path)
    finally:
        import torch, gc
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
    print(f"[DONE] Audio Remix created at: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Audio Remix.")
    parser.add_argument("input", help="Path to the input colorized video")
    parser.add_argument("original", help="original path with audio")
    parser.add_argument("output", help="Output path")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input file does not exist: {args.input}")
        exit(1)

    if not os.path.exists(args.original):
        print(f"[ERROR] original file does not exist: {args.original}")
        exit(1)

    try:
        final_output = remix_audio_cached(args.input, args.original, args.output)
        print(f"✔️ Audio Remixing completed. Final video at: {final_output}")
    except Exception as e:
        print(f"[ERROR] Audio Remixing failed: {e}")
        exit(1)