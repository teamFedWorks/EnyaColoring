import os

def delete_comfyui_pngs():
    # Get the folder where this script is located
    folder_path = os.path.dirname(os.path.abspath(__file__))

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.startswith("ComfyUI") and filename.endswith(".png"):
            file_path = os.path.join(folder_path, filename)
            try:
                print(f"Deleting file: {filename} ...")
                os.remove(file_path)
                print(f"✅ Deleted: {filename}")
            except Exception as e:
                print(f"❌ Could not delete {filename}: {e}")

if __name__ == "__main__":
    delete_comfyui_pngs()
