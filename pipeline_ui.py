import sys
import os
import runpy
import ipywidgets as widgets
from IPython.display import display, clear_output
from pathlib import Path

# Add your custom project path
sys.path.insert(0, "/workspace")
from Utils.main_utils import get_input_video_path

def launch_video_pipeline_ui():
    # --- Widget setup ---
    input_method_toggle = widgets.ToggleButtons(
        options=[("YouTube", "youtube"), ("Manual Path", "manual")],
        description="Input:"
    )
    url_input = widgets.Text(description='YouTube URL:', placeholder='https://...')
    file_input = widgets.Text(description='Local Path:', placeholder='input_videos/file.mp4')

    face_restore_toggle = widgets.ToggleButtons(
        options=[("Enable Face Restore", "True"), ("Disable Face Restore", "False")],
        description="Face Restore:"
    )
    upscale_toggle = widgets.ToggleButtons(
        options=[("Enable Upscale", "True"), ("Disable Upscale", "False")],
        description="Upscale:"
    )

    # NEW: QUALITY MODE TOGGLE
    quality_toggle = widgets.ToggleButtons(
        options=[("Quality", "quality"), ("Fast", "fast")],
        description="Mode:"
    )

    upscale_value_input = widgets.Text(
        value='2.0',
        placeholder='1.0 to 4.0',
        layout=widgets.Layout(width='150px'),
        disabled=False
    )
    upscale_value_row = widgets.HBox([
        widgets.Label(value='Upscale Value:', layout=widgets.Layout(width='120px')),
        upscale_value_input
    ])

    run_button = widgets.Button(description="Run Pipeline", button_style='success')
    output = widgets.Output()

    # --- Input toggle logic ---
    def on_input_method_change(change):
        method = change['new']
        url_input.disabled = (method != 'youtube')
        file_input.disabled = (method != 'manual')

    input_method_toggle.observe(on_input_method_change, names='value')
    on_input_method_change({'new': input_method_toggle.value})

    # --- Upscale toggle logic ---
    def on_upscale_toggle_change(change):
        if change['new'] == "True":
            upscale_value_row.layout.display = 'flex'
            upscale_value_input.disabled = False
        else:
            upscale_value_row.layout.display = 'none'
            upscale_value_input.disabled = True

    upscale_toggle.observe(on_upscale_toggle_change, names='value')
    on_upscale_toggle_change({'new': upscale_toggle.value})

    # --- Run logic ---
    def on_run_button_clicked(b):
        with output:
            clear_output()
            method = input_method_toggle.value
            youtube_url = url_input.value.strip() if method == 'youtube' else None
            manual_path = file_input.value.strip() if method == 'manual' else None

            try:
                video_path = get_input_video_path(youtube_url=youtube_url, manual_path=manual_path)
                unet_flag = "true"
                face_flag = face_restore_toggle.value
                upscale_flag = upscale_toggle.value
                upscale_value = upscale_value_input.value.strip()

                # NEW: Select pipeline file based on quality toggle
                selected_mode = quality_toggle.value
                pipeline_file = "pipeline.py" if selected_mode == "quality" else "pipeline_colorful.py"

                # Validate upscale value
                if upscale_flag == "True":
                    try:
                        val = float(upscale_value)
                        if not (1.0 <= val <= 4.0):
                            raise ValueError("Upscale value must be between 1.0 and 4.0")
                    except ValueError as ve:
                        print(f"⚠️ Invalid upscale value: {ve}")
                        return

                print(f"📥 Input: {video_path}")
                print(f"🎛️ U-Net Enabled   : {unet_flag}")
                print(f"🎭 Face Restore    : {face_flag}")
                print(f"⚡ Mode Selected    : {selected_mode} → Running {pipeline_file}")
                print(f"📈 Upscale Enabled : {upscale_flag}")
                if upscale_flag == "True":
                    print(f"🔢 Upscale Value   : {upscale_value}")
                print("🚀 Running pipeline...\n")

                # Run selected pipeline
                sys.argv = [pipeline_file, video_path, unet_flag, face_flag, upscale_flag, upscale_value, "False"]
                runpy.run_path(pipeline_file, run_name="__main__")

            except Exception as e:
                print("⚠️ Error:", str(e))

    run_button.on_click(on_run_button_clicked)

    # --- Show UI ---
    display(widgets.VBox([
        input_method_toggle,
        url_input,
        file_input,
        face_restore_toggle,
        upscale_toggle,
        upscale_value_row,
        quality_toggle,        # NEW MODE TOGGLE
        run_button,
        output
    ]))

# --- Entry ---
if __name__ == "__main__":
    try:
        get_ipython
        launch_video_pipeline_ui()
    except NameError:
        print("⚠️ Errors in pipeline.py")
