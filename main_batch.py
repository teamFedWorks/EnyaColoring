import pandas as pd
from datetime import datetime
import runpy
import sys
import smtplib
from email.message import EmailMessage
from Utils.main_utils import get_input_video_path
import os
import cv2
import subprocess
from datetime import datetime

sys.path.insert(0, "/workspace")

# --- CONFIGURATION ---
GOOGLE_SHEET_CSV_LINK = "https://docs.google.com/spreadsheets/d/1RB4YnVeaiJ7hDFf69k2VQ66JZJvdK06udk7KgdBm4cI/export?format=csv&gid=0"
MASTER_CSV_PATH = "master_sheet.csv"
EMAIL_FROM = "tvivekanand99@gmail.com"
EMAIL_TO = "tvivekanand99@gmail.com"
EMAIL_PASSWORD = "uwechomkqzltnzqd"
AUTOMATED = False

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

# --- Merge & Deduplicate Sheet ---
def update_master_csv():
    try:
        online_df = pd.read_csv(GOOGLE_SHEET_CSV_LINK)
        print(f"✅ Fetched {len(online_df)} rows from online sheet")

        for col in ["Executed", "Start Time", "End Time", "Remarks"]:
            if col not in online_df.columns:
                online_df[col] = ""

        try:
            local_df = pd.read_csv(MASTER_CSV_PATH)
            print(f"📂 Loaded {len(local_df)} rows from local master CSV")
        except FileNotFoundError:
            local_df = pd.DataFrame(columns=online_df.columns.tolist() + ["Input Path"])

        combined_df = pd.concat([local_df, online_df], ignore_index=True)

        if "Input Path" not in combined_df.columns:
            combined_df["Input Path"] = ""

        dupes = combined_df.duplicated(subset="YouTube URL", keep="first")
        if dupes.any():
            print(f"⚠️ Ignored {dupes.sum()} duplicate URLs from online sheet")

        combined_df = combined_df.drop_duplicates(subset="YouTube URL", keep="first")

        combined_df.to_csv(MASTER_CSV_PATH, index=False)
        print(f"✅ Updated and deduplicated master sheet: {len(combined_df)} rows")
        return combined_df

    except Exception as e:
        print(f"❌ Failed to update master sheet: {e}")
        send_email("❌ Sheet Merge Error", str(e))
        return pd.DataFrame()

# --- Run pipeline ---
def run_pipeline_entry(youtube_url, existing_path=None, df=None, index=None,
                       upscale_value="2.0", face_flag="False", upscale_flag="True", unet_flag="true"):
    try:
        import torch, gc
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()

        start_time = datetime.now().isoformat()

        if existing_path and isinstance(existing_path, str) and existing_path.strip() and os.path.exists(existing_path):
            video_path = existing_path
            print(f"📂 Reusing existing input path: {video_path}")
        else:
            video_path = get_input_video_path(youtube_url=youtube_url, manual_path=None)
            print(f"📥 Resolved Path: {video_path}")
            if df is not None and index is not None:
                df.at[index, "Input Path"] = video_path
                df.to_csv(MASTER_CSV_PATH, index=False)
                print("💾 Saved 'Input Path' to master CSV")


        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print("❌ Failed to open video for shape check.")
            else:
                width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps    = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps else "?"
                print(f"📏 Input Video Shape: {width}x{height}, {frame_count} frames at {fps} fps ({duration:.2f} sec)")
            cap.release()
        except Exception as e:
            print(f"⚠️ Error reading video shape: {e}")

        # --- Ask for flags here if not automated ---
        if not AUTOMATED:
            face_flag = input("🎭 Use face restoration? [True/False] (default: False): ").strip() or "False"
            upscale_flag = input("⬆️ Use background upscaling? [True/False] (default: True): ").strip() or "True"
            upscale_value = input("🔢 Upscale value [default: 2.0]: ").strip() or "2.0"
            unet_flag = input("🎨 Use U-Net colorization? [true/false] (default: true): ").strip() or "true"
            crop_flag = input("✂️ Crop this video before processing? [y/N]: ").strip().lower() == 'y'
        else:
            upscale_value = upscale_value or "2.0"
            face_flag = face_flag or "False"
            upscale_flag = upscale_flag or "True"
            unet_flag = unet_flag or "true"
            crop_flag = False


        # --- Optional Cropping ---
        if crop_flag:
            start_crop = input("⏱️ Enter start time (HH:MM:SS): ").strip()
            end_crop = input("⏱️ Enter end time (HH:MM:SS): ").strip()

            video_dir = os.path.dirname(video_path)
            temp_cropped_path = os.path.join(video_dir, "temp_cropped_video.mp4")

            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-ss", start_crop, "-to", end_crop,
                "-i", video_path,
                "-c", "copy",
                temp_cropped_path
            ]

            try:
                print(f"🔧 Cropping video from {start_crop} to {end_crop}...")
                subprocess.run(ffmpeg_cmd, check=True)
                os.replace(temp_cropped_path, video_path)  # Overwrite original
                print(f"✂️ Overwrote original video with cropped content: {video_path}")
            except subprocess.CalledProcessError as e:
                print(f"❌ ffmpeg failed: {e}")
                return start_time, "", "Error", f"FFmpeg crop failed: {e}", video_path
            except Exception as e:
                print(f"❌ Failed to overwrite video: {e}")
                return start_time, "", "Error", f"Overwrite failed: {e}", video_path

        print(f"\n🎬 Processing: {youtube_url}")
        sys.argv = ["pipeline.py", video_path, unet_flag, face_flag, upscale_flag, upscale_value]
        runpy.run_path("pipeline.py", run_name="__main__")

        end_time = datetime.now().isoformat()
        return start_time, end_time, "Yes", "", video_path

    except Exception as e:
        error_time = datetime.now().isoformat()
        import torch, gc
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        return error_time, "", "Error", str(e), video_path

# --- Main Process ---
def main():
    try:
        df = update_master_csv()
        if df.empty:
            print("🚫 No data to process.")
            return

        updated_rows = 0
        for i, row in df.iterrows():
            if str(row.get("Executed", "")).strip().lower() == "yes":
                continue

            youtube_url = str(row.get("YouTube URL", "") or "").strip()
            if not youtube_url.startswith("http"):
                print(f"⚠️ Invalid URL at row {i+1}")
                continue

            print(f"\n🎯 Video URL: {youtube_url}")
            if not AUTOMATED and input("➡️ Run pipeline for this video? [y/n]: ").strip().lower() != 'y':
                continue

            existing_path = str(row.get("Input Path", "") or "").strip()

            start_time, end_time, status, error, final_path = run_pipeline_entry(
                youtube_url,
                existing_path=existing_path,
                df=df,
                index=i
            )

            df.at[i, "Executed"] = status
            df.at[i, "Start Time"] = start_time
            df.at[i, "End Time"] = end_time or "N/A"
            df.at[i, "Remarks"] = str(error or "")
            df.at[i, "Input Path"] = final_path

            # Email notification
            if status == "Yes":
                subject = "✅ Pipeline Completed"
                body = f"Video processed successfully:\n\n{youtube_url}\nStart: {start_time}\nEnd: {end_time}"
            else:
                subject = "❌ Pipeline Failed"
                body = f"Pipeline failed:\n\n{youtube_url}\nStart: {start_time}\nError: {error}"

            send_email(subject, body)
            updated_rows += 1

        #if updated_rows:
            df.to_csv(MASTER_CSV_PATH, index=False)
            print(f"✅ Saved updated master CSV with {updated_rows} new run")
        #else:
        print("📌 No new videos to process.")

    except Exception as e:
        print("❌ Critical script error:", str(e))
        send_email("❌ Pipeline Script Crash", str(e))

# --- Entry ---
if __name__ == "__main__":
    main()
