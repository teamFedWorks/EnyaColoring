# update_sheet.py

import pandas as pd
from config import GOOGLE_SHEET_CSV_LINK, MASTER_CSV_PATH

def update_master_csv():
    try:
        # Load online sheet
        online_df = pd.read_csv(GOOGLE_SHEET_CSV_LINK, dtype=str)
        print(f"✅ Fetched {len(online_df)} rows from online sheet")

        # Ensure all required columns exist in the online sheet
        required_columns = [
            "YouTube URL", "Executed", "Start Time", "End Time", "Remarks",
            "face_upscale", "background_upscale", "background_upscale_value",
            "unet", "clahe", "crop_video", "start_time (00:00:00)", "end_time (01:22:33)"
        ]
        for col in required_columns:
            if col not in online_df.columns:
                online_df[col] = ""

        # Load local master CSV or initialize new one
        try:
            local_df = pd.read_csv(MASTER_CSV_PATH,  dtype=str)
            print(f"📂 Loaded {len(local_df)} rows from local master CSV")
        except FileNotFoundError:
            local_df = pd.DataFrame(columns=online_df.columns.tolist() + ["Input Path"])


        # Determine intersection columns, preserve order from online_df
        common_columns = [col for col in online_df.columns if col in local_df.columns]

        # Add 'Input Path' at the end explicitly if not already in online_df
        final_columns = common_columns + (["Input Path"] if "Input Path" in local_df.columns else [])

        # Reduce both to same columns
        online_df = online_df[common_columns]
        local_df = local_df[final_columns]

        # Ensure online_df also has 'Input Path' with empty values
        if "Input Path" in final_columns and "Input Path" not in online_df.columns:
            online_df["Input Path"] = ""

        # Reorder columns of online_df to match final_columns
        online_df = online_df[final_columns]

        # Merge online and local data
        combined_df = pd.concat([local_df, online_df], ignore_index=True)

        # Ensure local-use-only 'Input Path' column exists
        if "Input Path" not in combined_df.columns:
            combined_df["Input Path"] = ""

        # Drop duplicates by YouTube URL, keep first
        dupes = combined_df.duplicated(subset="YouTube URL", keep="first")
        if dupes.any():
            print(f"⚠️ Ignored {dupes.sum()} duplicate URLs")

        combined_df = combined_df.drop_duplicates(subset="YouTube URL", keep="first")

        # Save to master CSV
        combined_df.to_csv(MASTER_CSV_PATH, index=False)
        print(f"✅ Updated and saved master sheet: {len(combined_df)} rows")

        # Print first 10 rows of updated master sheet
       # print("\n🔎 Preview of first 10 rows:")
        # from tabulate import tabulate
        # print("\n🔎 Preview of first 10 rows:")
        # print(tabulate(combined_df.head(10), headers='keys', tablefmt='fancy_grid'))


    except Exception as e:
        print(f"❌ Error updating master CSV: {e}")

if __name__ == "__main__":
    update_master_csv()
