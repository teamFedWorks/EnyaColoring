import pandas as pd
from datetime import datetime
import runpy
import sys
import smtplib
from email.message import EmailMessage
import os
os.environ["OUTPUT_ROOT"] = "output_videos_1"
os.environ["INPUT_ROOT"] = "input_videos_1"
from Utils.main_utils import get_input_video_path
import cv2
import subprocess
from config import EMAIL_FROM, EMAIL_TO, EMAIL_PASSWORD, MASTER_CSV_PATH

sys.path.insert(0, "/workspace")

# --- Read config values ---
EMAIL_FROM = EMAIL_FROM
EMAIL_TO = EMAIL_TO
EMAIL_PASSWORD = EMAIL_PASSWORD

# --- Prompt automation and notification flags ---
automated_input = input("⚙️ Run in automated mode? [y/N]: ").strip().lower()
AUTOMATED = automated_input in ["y", "yes"]

notify_input = input("📨 Enable email notification on each video? [y/N]: ").strip().lower()
NOTIFY_ON_EACH = notify_input in ["y", "yes"]

# --- Validate config if notification enabled ---
if NOTIFY_ON_EACH:
    required_keys = [EMAIL_FROM, EMAIL_TO, EMAIL_PASSWORD]
    if not all(required_keys):
        raise ValueError("Missing required config values (email credentials).")

# --- Ask for CSV path ---
default_path = MASTER_CSV_PATH
csv_path_input = input(f"📁 Use default master CSV path '{default_path}'? [Y/n]: ").strip().lower()
MASTER_CSV_PATH = default_path if csv_path_input != 'n' else input("📝 Enter alternate CSV path: ").strip()
if not MASTER_CSV_PATH:
    raise ValueError("CSV path must be specified.")

# --- Email Utility ---
def send_email(subject, body):
    try:
        msg = EmailMessage()
        msg.set_content(body)
        msg["Subject"] = subject
        msg["From"] = EMAIL_FROM
        msg["To"] = EMAIL_TO
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_FROM, EMAIL_PASSWORD)
            smtp.send_message(msg)
        print(f"📧 Email sent: {subject}")
    except Exception as e:
        print(f"⚠️ Email failed: {e}")

# --- Run pipeline ---
def run_pipeline_entry(youtube_url, existing_path=None, df=None, index=None):
    try:
        import torch, gc
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()

        start_time = datetime.now().isoformat()

        if existing_path and os.path.exists(existing_path):
            video_path = existing_path
            print(f"📂 Reusing existing input path: {video_path}")
        else:
            video_path = get_input_video_path(youtube_url=youtube_url, manual_path=None)
            print(f"📥 Resolved Path: {video_path}")
            if df is not None and index is not None:
                df.at[index, "Input Path"] = video_path
                df.to_csv(MASTER_CSV_PATH, index=False)

        # Check shape
        try:
            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"📏 Shape: {width}x{height}, {frame_count} frames at {fps} fps")
            if(height>500):
                scale_default = "2"
            else:
                scale_default = "1"
            cap.release()
        except Exception as e:
            print(f"⚠️ Video shape error: {e}")

        row = df.loc[index]
        truthy_values = ["true", "1", "yes", "y", "t"]
        falsy_values = ["false", "0", "no", "n", "f"]
        if AUTOMATED:

        
            def safe_get(row, key, default):
                val = row.get(key)
                if pd.isna(val) or str(val).strip().lower() in ["", "nan", "na"]:
                    return default
                return str(val).strip()
        
            def validate_bool(value, key_name):
                lowered = value.lower()
                if lowered in truthy_values:
                    return True
                elif lowered in falsy_values:
                    return False
                else:
                    raise ValueError(
                        f"[Row {index}] Invalid boolean value for '{key_name}': '{value}'. "
                        f"Allowed values: {truthy_values + falsy_values}")
        
            row = df.loc[index]
        
            face_flag_raw = safe_get(row, "face_upscale", "False")
            face_flag = validate_bool(face_flag_raw, "face_upscale")
        
            upscale_flag_raw = safe_get(row, "background_upscale", "False")
            upscale_flag = validate_bool(upscale_flag_raw, "background_upscale")
        
            upscale_value = safe_get(row, "background_upscale_value", scale_default)
        
            unet_flag_raw = safe_get(row, "unet", "true")
            unet_flag = validate_bool(unet_flag_raw, "unet")

            clahe_flag_raw = safe_get(row, "clahe", "false")
            clahe_flag = validate_bool(clahe_flag_raw, "clahe")
        
            crop_flag_raw = safe_get(row, "crop_video", "false")
            crop_flag = validate_bool(crop_flag_raw, "crop_video")
        
            crop_start = safe_get(row, "start_time (00:00:00)", "")
            crop_end = safe_get(row, "end_time (01:22:33)", "")
   
        else:
            truthy_values = ["true", "1", "yes", "y", "t"]
            falsy_values = ["false", "0", "no", "n", "f"]
        
            def validate_bool(value, key_name):
                lowered = value.strip().lower()
                if lowered in truthy_values:
                    return True
                elif lowered in falsy_values:
                    return False
                else:
                    raise ValueError(
                        f"[Manual Input] Invalid boolean value for '{key_name}': '{value}'. "
                        f"Allowed values: {truthy_values + falsy_values}"
                    )
        
            def get_validated_input(prompt, key, default):
                while True:
                    raw = input(f"{prompt} ").strip() or default
                    try:
                        return validate_bool(raw, key)
                    except ValueError as e:
                        print(f"❌ {e}. Please try again.")
        
            face_flag = get_validated_input("🎭 Use face restoration? [True/False]:", "face_upscale", "False")
            upscale_flag = get_validated_input("⬆️ Use background upscaling? [True/False]:", "background_upscale", "True")
        
            upscale_value = input("🔢 Upscale value range [1,4]: ").strip() or "2.0"
        
            unet_flag = get_validated_input("🎨 Use U-Net? [true/false]:", "unet", "true")
            clahe_flag = get_validated_input("🎨 Use clahe? [true/false]:", "clahe", "false")
            crop_flag = get_validated_input("✂️ Crop video? [y/N]:", "crop_video", "false")
        
            crop_start = input("⏱️ Start time (HH:MM:SS): ").strip() if crop_flag else ""
            crop_end = input("⏱️ End time (HH:MM:SS): ").strip() if crop_flag else ""


        print("\n📌 Running pipeline with the following parameters:")
        print(f"   📽️  Video Path     : {video_path}")
        print(f"   🎭  Face Restore    : {face_flag}")
        print(f"   ⬆️  Background Upscale : {upscale_flag}")
        print(f"   🔢  Upscale Value   : {upscale_value}")
        print(f"   🧠  U-Net Flag      : {unet_flag}")
        print(f"   ✨  CLAHE Flag      : {clahe_flag}")
        print(f"   ✂️  Crop Applied   : {crop_flag}")
        if crop_flag:
            print(f"      Start Time     : {crop_start}")
            print(f"      End Time       : {crop_end}")

        if crop_flag and crop_start and crop_end:
            temp_cropped = os.path.join(os.path.dirname(video_path), "temp_crop.mp4")
            cmd = ["ffmpeg", "-y", "-ss", crop_start, "-to", crop_end, "-i", video_path, "-c", "copy", temp_cropped]
            subprocess.run(cmd, check=True)
            os.replace(temp_cropped, video_path)
            print(f"✂️ Cropped video replaced: {video_path}")


        sys.argv = ["pipeline.py", video_path, str(unet_flag), str(face_flag), str(upscale_flag), upscale_value, str(clahe_flag)]
        runpy.run_path("pipeline.py", run_name="__main__")

        return start_time, datetime.now().isoformat(), "Yes", "", video_path

    except Exception as e:
        import torch, gc
        torch.cuda.empty_cache()
        gc.collect()
        return datetime.now().isoformat(), "", "Error", str(e), existing_path or "?"

# --- Main ---
def main():
    df = pd.read_csv(MASTER_CSV_PATH)

    # Fix column dtypes
    required_columns = {
        "YouTube URL": "",
        "face_upscale": "",
        "background_upscale": "",
        "background_upscale_value": "",
        "unet": "",
        "clahe":"",
        "crop_video": "",
        "start_time (00:00:00)": "",
        "end_time (01:22:33)": "",
        "Executed": "",
        "Start Time": "",
        "End Time": "",
        "Remarks": "",
        "Input Path": ""
    }
    
    for col, default in required_columns.items():
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].astype(str)


    if df.empty:
        print("🚫 No data to process.")
        return

    updated_rows = 0
    for i, row in df.iterrows():
        if str(row.get("Executed", "")).strip().lower() == "yes":
            continue

        url = str(row.get("YouTube URL", "")).strip()
        if not url.startswith("http"):
            continue
            
        print(f"\n🎯 Video URL: {url}")
        if not AUTOMATED and input(f"▶️ Process this video? {url} [y/N]: ").strip().lower() != 'y':
            continue

        start, end, status, remark, path = run_pipeline_entry(
            youtube_url=url,
            existing_path=row.get("Input Path", ""),
            df=df,
            index=i
        )

        df.at[i, "Executed"] = status
        df.at[i, "Start Time"] = start
        df.at[i, "End Time"] = end or "N/A"
        df.at[i, "Remarks"] = remark
        #df.at[i, "Input Path"] = path

        if NOTIFY_ON_EACH:
            subject = "✅ Pipeline Completed" if status == "Yes" else "❌ Pipeline Failed"
            body = f"Video: {url}\nStart: {start}\nEnd: {end}\nStatus: {status}\nRemarks: {remark}"
            send_email(subject, body)

        updated_rows += 1
        df.to_csv(MASTER_CSV_PATH, index=False)
    
    print(f"✅ Processed {updated_rows} videos.")

# if __name__ == "__main__":
#     main()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Pipeline interrupted by user.")
        if NOTIFY_ON_EACH:
            send_email(
                "⚠️ Pipeline Interrupted",
                "The pipeline was manually interrupted by the user (KeyboardInterrupt)."
            )
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        if NOTIFY_ON_EACH:
            send_email(
                "❌ Pipeline Crashed",
                f"The pipeline encountered an error:\n{str(e)}"
            )
        raise
    finally:
        if NOTIFY_ON_EACH:
            send_email(
                "ℹ️ Pipeline Finished",
                "The pipeline script has stopped running (finished, crashed, or interrupted)."
            )

