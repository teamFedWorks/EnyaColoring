# -*- coding: utf-8 -*-
import sys
import os
import torch, gc
from pathlib import Path 

# ------------------------
# Take input video path as command-line argument
# ------------------------
if len(sys.argv) > 1:
    input_video_path = sys.argv[1]
    unet_flag = sys.argv[2].strip().lower() in ['true', '1', 'yes', 'y']
    face_restore_flag = sys.argv[3].strip().lower() in ['true', '1', 'yes', 'y']
    upscale_flag = sys.argv[4].strip().lower() in ['true', '1', 'yes', 'y']
    try:
        upscale_value = float(sys.argv[5])
        if not (1.0 <= upscale_value <= 4.0):
            raise ValueError("Upscale value must be between 1.0 and 4.0")
    except ValueError as ve:
        raise ValueError(f"Invalid upscale value: {ve}")
    clahe_flag = sys.argv[6].strip().lower() in ['true', '1', 'yes', 'y']
else:
    raise ValueError("Usage: pipeline.py <input_video_path>")

# ------------------------
# Utility to clear GPU memory
# ------------------------
def clear_gpu():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()



import sys
import os
import torch, gc
from pathlib import Path 
import subprocess
import time
import signal

def start_comfyui():
    print("🚀 Starting ComfyUI on port 8188...")
    log_file = open("/workspace/comfyui_runtime.log", "w")
    process = subprocess.Popen(
        ["python", "/opt/comfyui/main.py", "--listen", "0.0.0.0", "--port", "8188"],
        stdout=log_file,
        stderr=log_file,
        preexec_fn=os.setsid  # start new process group (so we can kill easily)
    )
    time.sleep(120)  # wait for it to initialize
    print(f"✅ ComfyUI started with PID {process.pid}")
    return process

def stop_comfyui(process):
    if process:
        print(f"🛑 Stopping ComfyUI (PID: {process.pid})...")
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait()
        print("✅ ComfyUI stopped.")

def stop_comfyui(process):
    if process:
        print(f"🛑 Stopping ComfyUI (PID: {process.pid})...")
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait(timeout=300)
        time.sleep(120)  # give it time to release GPU and port
        print("✅ ComfyUI stopped and cleaned.")





import requests

def wait_for_comfyui(port=8188, timeout=600):
    print(f"⏳ Waiting for ComfyUI to start on port {port}...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            r = requests.get(f"http://127.0.0.1:{port}")
            if r.status_code == 200:
                print("✅ ComfyUI is ready!")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(5)
    raise TimeoutError("❌ ComfyUI failed to start within timeout")



restored_video_path= None
faces_upscaled_video_path= None
background_upscaled_video_path = None
scene_split_preview_video_path = None
scene_split_preview_upscaled_video_path = None
deoldify_video_path = None
unet_video_path = None
flux_path = None
colorized_final_video_path = None
post_processed_video_path = None
final_video_path = None
final_postprocessed_video_path =None

# ------------------------
# Task 1: Restore B&W Film
# ------------------------
clear_gpu()
from Utils.main_utils import restore_bw_film_cached
input_path = input_video_path
restored_video_path = restore_bw_film_cached(input_path, input_video_path)
print(f"Restored video available at: {restored_video_path}")

# ------------------------
# Task 2: Face Enhancement
# ------------------------
clear_gpu()
from Utils.main_utils import upscale_faces_cached
# import io
# import contextlib
input_path = restored_video_path or input_video_path
if(face_restore_flag):
    #with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    faces_upscaled_video_path = upscale_faces_cached(input_path, input_video_path)
    print("Face-enhanced video available at:", faces_upscaled_video_path)

# ------------------------
# Task 3: Backgroung upscaling
# ------------------------
clear_gpu()
from Utils.main_utils import background_upscale_video_onnx_cached, downscale_video_in_place, resize_video_in_place
input_path = faces_upscaled_video_path or input_path
if(upscale_flag):
    #downscale_video_in_place(input_path, 2)
    model = 'models/Real-ESRGAN-General-x4v3.onnx'
    background_upscaled_video_path = background_upscale_video_onnx_cached(input_path, input_video_path, clahe_flag, scale = int(upscale_value), model_path=model)
    #resize_video_in_place(background_upscaled_video_path, input_video_path)
    print("Background Upscaled video at:", background_upscaled_video_path)



# ------------------------
# Task 4: Scene Split
# ------------------------
clear_gpu()
from Utils.main_utils import run_scene_split_cached

#input_path = faces_upscaled_video_path or restored_video_path or input_video_path
input_path = background_upscaled_video_path or input_path
scene_split_input_path = input_path
scene_split_preview_video_path = run_scene_split_cached(input_path, input_video_path, scale = int(upscale_value))
print("Scene split preview video available at:", scene_split_preview_video_path)



############################################# prev
# bw_path = Path(scene_split_preview_video_path)
# renamed_dir = bw_path.parent.parent / f"{bw_path.stem.replace('_bw','')}_images_prev"
# scene_refs_prev_name = f"{bw_path.stem.replace('_bw','')}_prevscene.mp4"
# scene_split_prevscene_video_path = os.path.join(renamed_dir, scene_refs_prev_name)

# print("Scene split PrevScene video available at:", scene_split_prevscene_video_path)

from pathlib import Path
import os

bw_path = Path(scene_split_preview_video_path)

# Replace "_images" with "_images_prev" in the parent folder name
renamed_dir = bw_path.parent.with_name(bw_path.parent.name.replace("_images", "_images_prev"))

# Remove ONLY the last "_bw" occurrence and append "_prevscene.mp4"
if bw_path.stem.endswith("_bw"):
    base_name = bw_path.stem[:-3]  # remove the last 3 characters "_bw"
else:
    base_name = bw_path.stem

scene_refs_prev_name = f"{base_name}_prevscene.mp4"

# Combine directory and filename
scene_split_prevscene_video_path = os.path.join(renamed_dir, scene_refs_prev_name)

print("Scene split PrevScene video available at:", scene_split_prevscene_video_path)





# ------------------------
# Task 5: Colorize Scenes Using Deoldify
# ------------------------
# clear_gpu()
# if not unet_flag:
#     from Utils.main_utils import deoldify_cached
#     #input_path = preview_video_path  or preview_upscaled_video
#     input_path = preview_upscaled_video or preview_video_path 
#     deoldify_video = deoldify_cached(input_path, input_video_path)
#     print("deoldify colored video at:", deoldify_video)


# ------------------------
# Task 5: Colorize Scenes Using Generator
# ------------------------
clear_gpu()
import subprocess

input_path = scene_split_preview_video_path 
# if unet_flag:   
#     generator_weights = "models/best_weights_epoch_0004.weights.h5"
#     from Utils.main_utils import run_unet_colorization_cached_subprocess
    
#     unet_video_path = run_unet_colorization_cached_subprocess(
#         input_bw_video=input_path,
#         unet_weights=generator_weights,
#         first_path=input_video_path
#     )

# comfy_process = start_comfyui()
# wait_for_comfyui()
clear_gpu()
from Utils.main_utils import comfyflux_colorize_video_cached, comfyflux_colorize_video_concat_cached, comfyflux_colorize_video_concat_scene_batch_cached
# import io
# import contextlib
input_path = scene_split_preview_video_path 
#prompt_ = "restore and colorize this without change in content, no warm/cool tint in entire image, color background, natural skintones"
prompt = "Restore and colorize. No warm/cool tint. Very Strictly Consistent color across dress or clothes per person. Natural skintones, color background."
flux_guidance=2.5
seed=2^24
#prompt_1 = "Restore and colorize with vivid colors, skin tones should be Natural, realistic. No warm/cool tint"
prompt_1 = "Restore and colorize this,  No warm/cool tint in entire image, color background, natural skintones"
flux_guidance=2.5
#flux_guidance=3.5
#flux_guidance=5
seed=469366467564800
#flux_path = comfyflux_colorize_video_cached(input_path, input_video_path, prompt_text = prompt_1, seed=seed, steps=10, cfg=1.0, flux_guidance=flux_guidance)
########concat_flux
# flux_guidance=5
# seed=2^24
# prompt_1 = "restore and colorize this, natural (light) skintones only, medium blue dresses only, no warm/cool tint in entire image, color background"
# flux_path  = comfyflux_colorize_video_concat_cached(
#     input_path,
#     input_video_path,
#     prompt_1,
#     seed=seed,
#     steps=10,
#     cfg=1.0,
#     flux_guidance=flux_guidance,
#     images_per_row=2,
#     total_images_per_combined=6
# )
########concat_flux_scene_batch
input_path = scene_split_prevscene_video_path
prompt_1 = "restore and colorize this, no warm/cool tint in entire image, color background, natural skintones"
prompt_1 = "restore and colorize this, no warm/cool tint in entire image, color background, natural and pale skintones"
prompt_1 = "restore and colorize this, no warm/cool tint in entire image, color background, natural and pale skintones, ornaments on people with gold color"
prompt_1 = "restore and colorize this, no warm/cool tint in entire image, color background, natural and pale skintones"
prompt_1 = "restore and colorize only skin"
seed=2^24
flux_guidance = 2.5
# flux_path  = comfyflux_colorize_video_concat_scene_batch_cached(
#     input_path,
#     input_video_path,
#     prompt_1,
#     seed=seed,
#     steps=20,
#     cfg=1.0,
#     flux_guidance=flux_guidance,
#     images_per_row=2,
#     total_images_per_combined=6
# )
# print("flux colorized video available at:", flux_path)
annotated_path = None
# stop_comfyui(comfy_process)

# if unet_flag:   
generator_weights = "models/best_weights_epoch_0004.weights.h5"
from Utils.main_utils import run_unet_colorization_cached_subprocess
input_path = scene_split_preview_video_path 
flux_path= run_unet_colorization_cached_subprocess(
    input_bw_video=input_path,
    unet_weights=generator_weights,
    first_path=input_video_path
)



#######prev for background
comfy_process = start_comfyui()
wait_for_comfyui()
clear_gpu()
from Utils.main_utils import comfyflux_colorize_video_cached
# import io
# import contextlib
input_path = scene_split_prevscene_video_path
print("prev input path", scene_split_prevscene_video_path)
#prompt_ = "restore and colorize this without change in content, no warm/cool tint in entire image, color background, natural skintones"
prompt = "Restore and colorize. No warm/cool tint. Very Strictly Consistent color across dress or clothes per person. Natural skintones, color background."
flux_guidance=2.5
seed=2^24
prompt_1 = "Restore and colorize with vivid colors, skin tones should be Natural, realistic. No warm/cool tint"
prompt_1 = "Restore and colorize this,  No warm/cool tint in entire image, color background, natural skintones"
# prompt_1 = "restore and colorize this, no warm/cool tint in entire image, color background, natural skintones, each person's dress differently (different color, natural, light, vivid)"


flux_guidance=3.5
flux_guidance=5
seed=469366467564800
seed=2^24
flux_prev_path = comfyflux_colorize_video_cached(input_path, input_video_path, prompt_text = prompt_1, seed=seed, steps=20, cfg=1.0, flux_guidance=flux_guidance)
print("flux colorized video available at:", flux_prev_path)
stop_comfyui(comfy_process)

#qwen

prompt_1 = (
    "restore and colorize with vivid colors, "
    "skin tones should be natural, realistic, "
    "and no warm or cool tint in the entire image"
)


############################################# prev
clear_gpu()
from Utils.main_utils import colorize_scenes_prev_cached
import sys
sys.path.insert(0, "/opt/deepex")
import sys
sys.path.insert(0, "/workspace")
input_path =  flux_prev_path 
colorized_final_video_prev_path =  colorize_scenes_prev_cached(scene_split_prevscene_video_path,  scene_split_input_path , input_path, input_video_path)
print("Colorized prev final video at:", colorized_final_video_prev_path)
org_flux_path = flux_path
flux_path = colorized_final_video_prev_path


#yolo mask replace
from Utils.main_utils import replace_masked_regions_between_videos
flux_path = replace_masked_regions_between_videos(org_flux_path, flux_path, output_suffix="_maskedmerge.mp4")
############################################# 


################




# from Utils.yolo_sam_deoldify_4_masks import process_video_cached
# input_path = flux_path 
# annotated_path =None
# # # Simple pipeline use (only main file returned)
# # output_path = run_deoldify_yolo_cached("input_videos/flux_video.mp4")
# # print("Main output:", output_path)

# # If you also want the YOLO debug file
# annotated_path =  process_video_cached(input_path)
# print("Main output:", annotated_path)




# from Utils.yolo_deoldify import run_deoldify_yolo_cached, run_deoldify_yolo
# input_path = flux_path 
# annotated_path =None
# # # Simple pipeline use (only main file returned)
# # output_path = run_deoldify_yolo_cached("input_videos/flux_video.mp4")
# # print("Main output:", output_path)

# # If you also want the YOLO debug file
# annotated_path = run_deoldify_yolo_cached(input_path)
# print("Main output:", annotated_path)
# #print("YOLO debug:", yolo_debug)


# clear_gpu()
# input_path = unet_video_path
# from Utils.main_utils import enhance_unet_cached
# enhanced_unet_path = enhance_unet_cached(input_path, input_video_path)
# print("Diffusion enhanced video available at:", enhanced_unet_path)

# import subprocess
# import os

# env = os.environ.copy()
# input_path = unet_video_path or scene_split_preview_video_path
# env["CUSTOM_INPUT_PATH"] = input_path          # or .mp4

# # Generate output path based on input
# base_dir = os.path.dirname(input_path)
# base_name = os.path.basename(input_path)
# output_name = "output_ge5_500_1_" + base_name
# output_path = os.path.join(base_dir, output_name)

# env["CUSTOM_OUTPUT_PATH"] = output_path                    # folder to save final result
# env["CUSTOM_PROMPT"] = "Colorize each person's dress differently (different color, natural, light, vivid), consistent per dress, natural skin tones, light vivid colors in background, no warm/cool tint"
# #env["CUSTOM_PROMPT"] = "turn image into anime style"
# env["CUSTOM_PROMPT"] = "restore and colorize this, no warm/cool tint in entire image, color background, natural skintones, each person's dress differently (different color, natural, light, vivid)"

# # env["CUSTOM_PROMPT"]  = "restore and colorize this, no warm/cool tint in entire image, color background, natural skintones, each person's dress (different color per person,  consistently same color entire dress, natural, light, vivid)"

# #env["CUSTOM_PROMPT"] = "colorize this with natural colors"
# #subprocess.run(["python", "/opt/comfyui/workflow_api.py"], env=env)
# unet_video_path = output_path
# #"Colorize with cinematic tones, not warm or cool" 


# ------------------------
# Task 6: Final Scene-wise Colorization Merge
# ------------------------
clear_gpu()
from Utils.main_utils import colorize_scenes_cached
import sys
sys.path.insert(0, "/opt/deepex")
import sys
sys.path.insert(0, "/workspace")
input_path =  annotated_path or flux_path or unet_video_path
colorized_final_video_path =  colorize_scenes_cached(scene_split_preview_video_path,  scene_split_input_path , input_path, input_video_path)
print("Colorized final video at:", colorized_final_video_path)








# ------------------------
# Task 7: Postprocess Videos
# ------------------------


clear_gpu()
from Utils.main_utils import postprocess_videos_cached
input_path = colorized_final_video_path
post_processed_video_path = postprocess_videos_cached(input_path, input_video_path)
print("postprocessed video at:", post_processed_video_path)

# ------------------------
# Task 8: Remix
# ------------------------
clear_gpu()
from Utils.main_utils import remix_audio_cached
input_path = post_processed_video_path
final_postprocessed_video_path = remix_audio_cached(input_path, input_video_path, "final_post_process")
print("final postprocessed video at:", final_postprocessed_video_path)




clear_gpu()

input_path = colorized_final_video_path
final_video_path = remix_audio_cached(input_path , input_video_path, "final_without_post_process")
print("final video at:", final_video_path)

