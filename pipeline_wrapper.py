#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Wrapper script for pipeline.py that handles YouTube URLs and manual paths.
This script should be placed on each remote node in the same directory as pipeline.py.

Note: CSV batch processing should use youtube_urls_batch_run.py directly, not this wrapper.
"""

import sys
import os
import subprocess

# Add workspace to path
sys.path.insert(0, "/workspace")

from Utils.main_utils import get_input_video_path

def main():
    if len(sys.argv) < 2:
        print("Usage: pipeline_wrapper.py <youtube_url|manual_path> [unet_flag] [face_restore_flag] [upscale_flag] [upscale_value] [clahe_flag] [quality_mode]")
        print("\nNote: For CSV batch processing, use youtube_urls_batch_run.py directly.")
        sys.exit(1)
    
    input_arg = sys.argv[1]
    
    # Check if input is a CSV file
    if input_arg.lower().endswith('.csv') or (os.path.exists(input_arg) and input_arg.lower().endswith('.csv')):
        print("ERROR: CSV files are not supported by pipeline_wrapper.py", file=sys.stderr)
        print("For batch CSV processing, use youtube_urls_batch_run.py directly.", file=sys.stderr)
        print(f"Received CSV path: {input_arg}", file=sys.stderr)
        sys.exit(1)
    
    # Determine if it's a YouTube URL or manual path
    is_youtube = 'youtube.com' in input_arg or 'youtu.be' in input_arg
    
    try:
        if is_youtube:
            # Download video from YouTube
            print(f"Downloading video from YouTube: {input_arg}")
            video_path = get_input_video_path(youtube_url=input_arg, manual_path=None)
        else:
            # Use manual path
            # Check if path exists and is a file (not a directory)
            if os.path.exists(input_arg):
                if os.path.isdir(input_arg):
                    raise ValueError(f"Input path is a directory, not a file: {input_arg}")
                if not os.path.isfile(input_arg):
                    raise ValueError(f"Input path exists but is not a regular file: {input_arg}")
            
            video_path = get_input_video_path(youtube_url=None, manual_path=input_arg)
        
        if not video_path or not os.path.exists(video_path):
            raise ValueError(f"Video path not found: {video_path}")
        
        # Verify it's actually a video file (basic check)
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v']
        if not any(video_path.lower().endswith(ext) for ext in video_extensions):
            print(f"WARNING: File extension doesn't look like a video file: {video_path}", file=sys.stderr)
        
        print(f"Using video path: {video_path}")
        
        # Determine pipeline file based on quality mode (default to quality)
        quality_mode = sys.argv[7] if len(sys.argv) > 7 else 'quality'
        pipeline_file = 'pipeline.py' if quality_mode == 'quality' else 'pipeline_colorful.py'
        print(f"⚡ Mode: {quality_mode} → Using {pipeline_file}")
        
        # Build pipeline command (exclude quality_mode from pipeline args, it's only for file selection)
        pipeline_args = [sys.executable, pipeline_file, video_path] + sys.argv[2:7]
        
        # Change to workspace directory
        os.chdir('/workspace')
        
        # Execute pipeline.py and stream output
        process = subprocess.Popen(
            pipeline_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream output in real-time
        for line in process.stdout:
            print(line, end='', flush=True)
        
        # Wait for process to complete
        process.wait()
        
        # Exit with pipeline's exit code
        sys.exit(process.returncode)
        
    except Exception as e:
        print(f"Error in wrapper: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
