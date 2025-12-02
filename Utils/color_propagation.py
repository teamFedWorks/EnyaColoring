import cv2
import os
import tempfile
import shutil
import numpy as np
import re
from Utils.main_utils import repair_video_file
from tqdm import tqdm
def reverse_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video for reversing: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    frames.reverse()

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for f in frames:
        out.write(f)
    out.release()



def blend_colorized_frames_lab(frame1, frame2, frame3):
    """
    Blend 3 BGR frames in LAB space:
    - Lightness is averaged.
    - A and B are chosen from the frame with strongest chroma per pixel.
    """
    # Convert to LAB and shift A/B to signed range [-128, 127]
    lab1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2LAB).astype(np.float32)
    for lab in (lab1, lab2, lab3):
        lab[:, :, 1:] -= 128.0

    # Split LAB channels
    L1, A1, B1 = cv2.split(lab1)
    L2, A2, B2 = cv2.split(lab2)
    L3, A3, B3 = cv2.split(lab3)

    # Average lightness
    L_blend = (L1 + L2 + L3) / 3

    # Compute chroma magnitude per frame
    C1 = np.sqrt(A1**2 + B1**2)
    C2 = np.sqrt(A2**2 + B2**2)
    C3 = np.sqrt(A3**2 + B3**2)

    # Stack and get index of max chroma
    chroma_stack = np.stack([C1, C2, C3])
    A_stack = np.stack([A1, A2, A3])
    B_stack = np.stack([B1, B2, B3])
    idx = np.argmax(chroma_stack, axis=0)

    # Select A and B from the most saturated frame
    A_blend = np.take_along_axis(A_stack, idx[None, ...], axis=0)[0]
    B_blend = np.take_along_axis(B_stack, idx[None, ...], axis=0)[0]

    # Merge, shift A/B back to [0, 255], convert to uint8
    lab_blend = cv2.merge([L_blend, A_blend + 128.0, B_blend + 128.0])
    lab_blend = np.clip(lab_blend, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab_blend, cv2.COLOR_LAB2BGR)







def colorize_scenes_basic(scene_refs_video,  before_path, colorized_scene_path, final_output_path):
    
    preview_dir = os.path.dirname(os.path.dirname(scene_refs_video)) 
    
    #final_output_path = get_cached_file(colorized_scene_path, "colored_full_video", video_source_path=first_path)
    base_video_name = os.path.splitext(os.path.basename(before_path))[0]
  
    if os.path.exists(final_output_path):
        print(f"[CACHE] Full Colorized output already exists: {final_output_path}")
        return str(final_output_path)
    from colorize_video import colorize_video_main, load_colorization_models
    opt, nonlocal_net, colornet, vggnet = load_colorization_models()

    
    color_scene_dir = os.path.dirname(final_output_path)
    image_dir = os.path.join(preview_dir, f"{base_video_name}_images")
    reference_video = colorized_scene_path
    
    output_dir = os.path.join(color_scene_dir, "colored_scenes")
    os.makedirs(output_dir, exist_ok=True)

    scene_files = sorted([
        f for f in os.listdir(preview_dir)
        if f.endswith(".mp4") and f.startswith(f"{base_video_name}-Scene")
    ])

    # Extract reference frames from preview_dir/ref_*.png
    cap = cv2.VideoCapture(reference_video)
    reference_frames = []
    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ref_path = os.path.join(color_scene_dir, f"ref_{frame_index:03d}.png")
        cv2.imwrite(ref_path, frame)
        reference_frames.append(ref_path)
        frame_index += 1
    cap.release()

    expected_ref_frames = 3 * len(scene_files)
    if len(reference_frames) != expected_ref_frames:
        raise ValueError(f"Expected {expected_ref_frames} reference frames, got {len(reference_frames)}")

    print(f"\n🎨 Starting colorization of {len(scene_files)} scenes with 3 refs each...\n")

    for idx, scene_file in enumerate(scene_files):
        scene_path = os.path.join(preview_dir, scene_file)
        scene_base = os.path.splitext(scene_file)[0]
        out_path_1 = os.path.join(output_dir, f"{scene_base}_1.mp4")
        out_path_2 = os.path.join(output_dir, f"{scene_base}_2.mp4")
        out_path_3 = os.path.join(output_dir, f"{scene_base}_3.mp4")

        ref_1 = reference_frames[3 * idx]
        ref_2 = reference_frames[3 * idx + 1]
        ref_3 = reference_frames[3 * idx + 2]

        print(f"🎬 Scene {idx + 1}/{len(scene_files)} - {scene_file}")



        # Forward colorization with first reference (_1)
        colorize_video_main(scene_path, ref_1, out_path_1, opt, nonlocal_net, colornet, vggnet)

        # Forward colorization with middle reference (_2)
        colorize_video_main(scene_path, ref_2, out_path_2, opt, nonlocal_net, colornet, vggnet)

        # Reverse colorization with third reference (_3)
        reversed_path = os.path.join(preview_dir, f"{scene_base}_reversed.mp4")
        temp_b_reversed = os.path.join(output_dir, f"{scene_base}_temp_reversed.mp4")
        reverse_video(scene_path, reversed_path)
        colorize_video_main(reversed_path, ref_3, temp_b_reversed, opt, nonlocal_net, colornet, vggnet)
        reverse_video(temp_b_reversed, out_path_3)
        os.remove(reversed_path)
        os.remove(temp_b_reversed)

    print(f"\n✅ All colorized clips saved to: {output_dir}")

    # Final blending step
    scene_files_1 = sorted([f for f in os.listdir(output_dir) if f.endswith("_1.mp4")])
    scene_files_2 = sorted([f for f in os.listdir(output_dir) if f.endswith("_2.mp4")])
    scene_files_3 = sorted([f for f in os.listdir(output_dir) if f.endswith("_3.mp4")])

    if not (len(scene_files_1) == len(scene_files_2) == len(scene_files_3)):
        raise RuntimeError("Mismatch in number of colorized scene files (_1, _2, _3)")

    cap = cv2.VideoCapture(os.path.join(output_dir, scene_files_1[0]))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(final_output_path, fourcc, fps, (width, height))

    #print(f"🎥 Blending and combining {len(scene_files_1)} scenes...")
    #tqdm.write(f"🎥 Blending and combining {len(scene_files_1)} scenes...")

    for idx, (file_1, file_2, file_3) in enumerate(zip(scene_files_1, scene_files_2, scene_files_3)):


        print(f"🔀 Blending Scene {idx+1}: {file_1} + {file_2} +{file_3}")
        #tqdm.write(f"🔀 Blending Scene {idx+1}: {file_1} + {file_2} +{file_3}")
        cap1 = cv2.VideoCapture(os.path.join(output_dir, file_1))
        cap2 = cv2.VideoCapture(os.path.join(output_dir, file_2))
        cap3 = cv2.VideoCapture(os.path.join(output_dir, file_3))

        total_frames = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))

        with tqdm(total=total_frames, desc=f"📽️ Scene {idx} frames") as pbar:
            while True:
                ret1, f1 = cap1.read()
                ret2, f2 = cap2.read()
                ret3, f3 = cap3.read()
    
                if not (ret1 and ret2 and ret3):
                    break
    
                blended = blend_colorized_frames_lab(f1, f2, f3)
                out.write(blended)
                pbar.update(1)

        cap1.release()
        cap2.release()
        cap3.release()

    out.release()
    print(f"\n✅ Final blended video saved to: {final_output_path}")
    del opt, nonlocal_net, colornet, vggnet
    repair_video_file(final_output_path)
    return final_output_path


def colorize_scenes(scene_refs_video, before_path, colorized_scene_path, final_output_path):
    import cv2, os
    from colorize_video import colorize_video_main, load_colorization_models
    from tqdm import tqdm
    import numpy as np
    from Utils.main_utils import repair_video_file

    preview_dir = os.path.dirname(os.path.dirname(scene_refs_video))
    base_video_name = os.path.splitext(os.path.basename(before_path))[0]

    if os.path.exists(final_output_path):
        print(f"[CACHE] Final blended output already exists: {final_output_path}")
        return final_output_path

    opt, nonlocal_net, colornet, vggnet = load_colorization_models()
    color_scene_dir = os.path.dirname(final_output_path)
    output_dir = os.path.join(color_scene_dir, "colored_scenes")
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load scene files
    scene_files = sorted([
        f for f in os.listdir(preview_dir)
        if f.endswith(".mp4") and f.startswith(f"{base_video_name}-Scene")
    ])

    # Step 2: Extract reference frames
    ref_frames = []
    cap = cv2.VideoCapture(colorized_scene_path)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ref_path = os.path.join(color_scene_dir, f"ref_{idx:03d}.png")
        cv2.imwrite(ref_path, frame)
        ref_frames.append(ref_path)
        idx += 1
    cap.release()

    expected_refs = 3 * len(scene_files)
    if len(ref_frames) != expected_refs:
        raise ValueError(f"Expected {expected_refs} reference frames, got {len(ref_frames)}")

    print(f"\n🎨 Starting colorization of {len(scene_files)} scenes...\n")

    # Step 3: Colorize scenes
    for idx, scene_file in enumerate(scene_files):
        scene_path = os.path.join(preview_dir, scene_file)
        scene_base = os.path.splitext(scene_file)[0]
        out1 = os.path.join(output_dir, f"{scene_base}_1.mp4")
        out2 = os.path.join(output_dir, f"{scene_base}_2.mp4")
        out3 = os.path.join(output_dir, f"{scene_base}_3.mp4")

        if all(os.path.exists(p) for p in [out1, out2, out3]):
            print(f"[CACHE] Skipping colorization for: {scene_file}")
            continue

        print(f"🎬 Colorizing Scene {idx+1}/{len(scene_files)}: {scene_file}")
        ref1, ref2, ref3 = ref_frames[3*idx : 3*idx+3]
        reversed_path = os.path.join(preview_dir, f"{scene_base}_reversed.mp4")
        temp_reversed = os.path.join(output_dir, f"{scene_base}_temp_reversed.mp4")

        colorize_video_main(scene_path, ref1, out1, opt, nonlocal_net, colornet, vggnet)
        colorize_video_main(scene_path, ref2, out2, opt, nonlocal_net, colornet, vggnet)
        reverse_video(scene_path, reversed_path)
        colorize_video_main(reversed_path, ref3, temp_reversed, opt, nonlocal_net, colornet, vggnet)
        reverse_video(temp_reversed, out3)
        os.remove(reversed_path)
        os.remove(temp_reversed)

    print("\n🎞️ Starting per-scene blending...\n")

    blended_scene_paths = []
    all_blended_exist = True

    for idx, scene_file in enumerate(scene_files):
        scene_base = os.path.splitext(scene_file)[0]
        out1 = os.path.join(output_dir, f"{scene_base}_1.mp4")
        out2 = os.path.join(output_dir, f"{scene_base}_2.mp4")
        out3 = os.path.join(output_dir, f"{scene_base}_3.mp4")
        blended_path = os.path.join(output_dir, f"{scene_base}_blended.mp4")

        if os.path.exists(blended_path):
            print(f"[CACHE] Skipping blending for: {scene_base}")
            blended_scene_paths.append(blended_path)
            continue

        all_blended_exist = False
        print(f"🔀 Blending Scene {idx+1}: {scene_base}")
        cap1 = cv2.VideoCapture(out1)
        cap2 = cv2.VideoCapture(out2)
        cap3 = cv2.VideoCapture(out3)

        fps = cap1.get(cv2.CAP_PROP_FPS)
        width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(blended_path, fourcc, fps, (width, height))

        with tqdm(total=total_frames, desc=f"Blending {scene_base}") as pbar:
            while True:
                r1, f1 = cap1.read()
                r2, f2 = cap2.read()
                r3, f3 = cap3.read()
                if not (r1 and r2 and r3):
                    break
                blended = blend_colorized_frames_lab(f1, f2, f3)
                writer.write(blended)
                pbar.update(1)

        cap1.release()
        cap2.release()
        cap3.release()
        writer.release()
        blended_scene_paths.append(blended_path)

    # Step 4: Final video generation
    if os.path.exists(final_output_path):
        print(f"[CACHE] Final output already exists: {final_output_path}")
        return final_output_path

    print(f"\n📦 Combining {len(blended_scene_paths)} blended scenes...\n")

    cap = cv2.VideoCapture(blended_scene_paths[0])
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    final_writer = cv2.VideoWriter(final_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for blended_scene in tqdm(blended_scene_paths, desc="🧩 Merging scenes"):
        cap = cv2.VideoCapture(blended_scene)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            final_writer.write(frame)
        cap.release()

    final_writer.release()
    repair_video_file(final_output_path)
    print(f"✅ Final blended video saved to: {final_output_path}")
    return final_output_path




#colorize_scenes_m
def colorize_scenes(scene_refs_video, before_path, colorized_scene_path, final_output_path):
    print_flag = False
    import cv2, os
    from colorize_video import colorize_video_main, load_colorization_models
    from tqdm import tqdm
    from Utils.main_utils import repair_video_file

    preview_dir = os.path.dirname(os.path.dirname(scene_refs_video))
    base_video_name = os.path.splitext(os.path.basename(before_path))[0]

    if os.path.exists(final_output_path):
        print(f"[CACHE] Final output already exists: {final_output_path}")
        return final_output_path

    opt, nonlocal_net, colornet, vggnet = load_colorization_models()
    color_scene_dir = os.path.dirname(final_output_path)
    output_dir = os.path.join(color_scene_dir, "colored_scenes")
    os.makedirs(output_dir, exist_ok=True)

    # Load scene files
    scene_files = sorted([
        f for f in os.listdir(preview_dir)
        if f.endswith(".mp4") and f.startswith(f"{base_video_name}-Scene")
    ])

    # Extract middle reference frames only
    ref_frames = []
    cap = cv2.VideoCapture(colorized_scene_path)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ref_path = os.path.join(color_scene_dir, f"ref_{idx:03d}.png")
        cv2.imwrite(ref_path, frame)
        ref_frames.append(ref_path)
        idx += 1
    cap.release()

    expected_refs = len(scene_files)  # One frame per scene now
    if len(ref_frames) != expected_refs:
        raise ValueError(f"Expected {expected_refs} reference frames, got {len(ref_frames)}")

    print(f"\n🎨 Starting colorization of {len(scene_files)} scenes using middle frame references...\n")

    colorized_scene_paths = []

    #for idx, scene_file in enumerate(scene_files):
    from tqdm import tqdm
    for idx, scene_file in enumerate(tqdm(scene_files, desc="🎬 Colorizing Scenes")):
        scene_path = os.path.join(preview_dir, scene_file)
        scene_base = os.path.splitext(scene_file)[0]
        colorized_path = os.path.join(output_dir, f"{scene_base}_colorized.mp4")

        if os.path.exists(colorized_path):
            print(f"[CACHE] Skipping colorization for: {scene_file}")
            colorized_scene_paths.append(colorized_path)
            continue

        if(print_flag):
           print(f"🎬 Colorizing Scene {idx+1}/{len(scene_files)}: {scene_file}")
        ref = ref_frames[idx]

        colorize_video_main(scene_path, ref, colorized_path, opt, nonlocal_net, colornet, vggnet)

        colorized_scene_paths.append(colorized_path)

    # Combine all colorized scenes directly (no blending)
    print(f"\n📦 Combining {len(colorized_scene_paths)} colorized scenes...\n")

    cap = cv2.VideoCapture(colorized_scene_paths[0])
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    final_writer = cv2.VideoWriter(final_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for colorized_scene in tqdm(colorized_scene_paths, desc="🧩 Merging scenes"):
        cap = cv2.VideoCapture(colorized_scene)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            final_writer.write(frame)
        cap.release()

    final_writer.release()
    repair_video_file(final_output_path)
    print(f"✅ Final colorized video saved to: {final_output_path}")
    return final_output_path



def colorize_scenes_prev(scene_refs_video, before_path, colorized_scene_path, final_output_path):
    print_flag =False
    import cv2, os
    from colorize_video import colorize_video_main, load_colorization_models
    from tqdm import tqdm
    from Utils.main_utils import repair_video_file

    preview_dir = os.path.dirname(os.path.dirname(scene_refs_video))
    base_video_name = os.path.splitext(os.path.basename(before_path))[0]

    if os.path.exists(final_output_path):
        print(f"[CACHE] Final output already exists: {final_output_path}")
        return final_output_path

    opt, nonlocal_net, colornet, vggnet = load_colorization_models()
    color_scene_dir = os.path.dirname(final_output_path)
    output_dir = os.path.join(color_scene_dir, "colored_scenes")
    os.makedirs(output_dir, exist_ok=True)

    # Load scene files
    # scene_files = sorted([
    #     f for f in os.listdir(preview_dir)
    #     if f.endswith(".mp4") and f.startswith("PrevScene")
    # ])

    scene_files = sorted(
        [f for f in os.listdir(preview_dir)
         if f.endswith(".mp4") and f.startswith("PrevScene")],
        key=lambda x: int(''.join(filter(str.isdigit, x)))
    )


    # Extract middle reference frames only
    ref_frames = []
    cap = cv2.VideoCapture(colorized_scene_path)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ref_path = os.path.join(color_scene_dir, f"ref_{idx:03d}.png")
        cv2.imwrite(ref_path, frame)
        ref_frames.append(ref_path)
        idx += 1
    cap.release()

    expected_refs = len(scene_files)  # One frame per scene now
    if len(ref_frames) != expected_refs:
        raise ValueError(f"Expected {expected_refs} reference frames, got {len(ref_frames)}")

    print(f"\n🎨 Starting colorization of {len(scene_files)} scenes using middle frame references...\n")

    colorized_scene_paths = []

    #for idx, scene_file in enumerate(scene_files):
    from tqdm import tqdm
    for idx, scene_file in enumerate(tqdm(scene_files, desc="🎬 Colorizing Scenes")):
        scene_path = os.path.join(preview_dir, scene_file)
        scene_base = os.path.splitext(scene_file)[0]
        colorized_path = os.path.join(output_dir, f"{scene_base}_colorized.mp4")

        if os.path.exists(colorized_path):
            print(f"[CACHE] Skipping colorization for: {scene_file}")
            colorized_scene_paths.append(colorized_path)
            continue

        if(print_flag):
           print(f"🎬 Colorizing Scene {idx+1}/{len(scene_files)}: {scene_file}")
        ref = ref_frames[idx]

        colorize_video_main(scene_path, ref, colorized_path, opt, nonlocal_net, colornet, vggnet)

        colorized_scene_paths.append(colorized_path)

    # Combine all colorized scenes directly (no blending)
    print(f"\n📦 Combining {len(colorized_scene_paths)} colorized scenes...\n")

    cap = cv2.VideoCapture(colorized_scene_paths[0])
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    final_writer = cv2.VideoWriter(final_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for colorized_scene in tqdm(colorized_scene_paths, desc="🧩 Merging scenes"):
        cap = cv2.VideoCapture(colorized_scene)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            final_writer.write(frame)
        cap.release()

    final_writer.release()
    repair_video_file(final_output_path)
    print(f"✅ Final colorized video saved to: {final_output_path}")
    return final_output_path




import os
import cv2
import re
from tqdm import tqdm

def colorize_actor_videos(actor_root_folder, reference_video_path, output_root_folder,
                         fps=30, prefix="Actor"):
    """
    actor_root_folder: path containing Scene_X/Actor_Y.mp4 videos
    reference_video_path: the middle frames video corresponding to each actor
    output_root_folder: where colorized actor videos will be saved
    colorize_func: callable like colorize_video_main(input_video_path, reference_file, output_video_path, ...)
    fps: FPS to process
    """
    from colorize_video import colorize_video_main, load_colorization_models
    opt, nonlocal_net, colornet, vggnet = load_colorization_models()
    os.makedirs(output_root_folder, exist_ok=True)

    # Step 1: Sort actor videos scene -> actor
    def sort_key(path):
        scene_match = re.search(r"Scene_(\d+)", path)
        actor_match = re.search(rf"{prefix}_(\d+)\.mp4$", path)
        scene_num = int(scene_match.group(1)) if scene_match else 999
        actor_num = int(actor_match.group(1)) if actor_match else 999
        return (scene_num, actor_num)

    # Collect actor videos
    actor_videos = []
    for root, dirs, files in os.walk(actor_root_folder):
        for f in files:
            if re.match(rf"{prefix}_\d+\.mp4$", f):
                actor_videos.append(os.path.join(root, f))

    if not actor_videos:
        print("No actor videos found.")
        return

    actor_videos = sorted(actor_videos, key=sort_key)
    print("Actor videos in processing order:")
    for v in actor_videos:
        print(v)

    # Step 2: Read reference video frames (one frame per actor)
    ref_cap = cv2.VideoCapture(reference_video_path)
    ref_frames = []
    while True:
        ret, frame = ref_cap.read()
        if not ret:
            break
        ref_frames.append(frame)
    ref_cap.release()

    if len(ref_frames) != len(actor_videos):
        print(f"⚠️ Warning: {len(ref_frames)} reference frames but {len(actor_videos)} actor videos.")
        min_len = min(len(ref_frames), len(actor_videos))
        actor_videos = actor_videos[:min_len]
        ref_frames = ref_frames[:min_len]

    # Step 3: Run colorize_video_main for each actor video
    for vid_path, ref_frame in tqdm(zip(actor_videos, ref_frames), desc="Colorizing actors", total=len(actor_videos)):
        actor_name = os.path.splitext(os.path.basename(vid_path))[0]
        output_path = os.path.join(output_root_folder, f"{actor_name}_colorized.mp4")

        # Save reference frame temporarily to pass to colorize_video_main
        temp_ref_path = os.path.join(output_root_folder, f"{actor_name}_ref.jpg")
        cv2.imwrite(temp_ref_path, ref_frame)

        # Call your colorize function
        # colorize_video_main(
        #     input_video_path=vid_path,
        #     reference_file=temp_ref_path,
        #     output_video_path=output_path, opt, nonlocal_net, colornet, vggnet)
        colorize_video_main(
            vid_path,
            temp_ref_path,
            output_path, opt, nonlocal_net, colornet, vggnet)

        os.remove(temp_ref_path)  # cleanup

    print("✅ All actor videos processed.")


# Example usage:
# colorize_actor_videos(
#     actor_root_folder="output_videos_latest/t8/output_scenes/output_actors",
#     reference_video_path="output_videos_latest/t8/final_middle_frames.mp4",
#     output_root_folder="output_videos_latest/t8/colorized_actors",
#     colorize_func=colorize_video_main
# )



def colorize_scenes_cached(scene_refs_video,  before_path, colorized_scene_path, output_path):
    """Propagating colors, using filecache to avoid reprocessing."""
    if os.path.exists(output_path):
        print(f"[CACHE] Propagated colored full video found: {output_path}")
        return output_path
    print("[INFO] Running Color propagation...")
    from Utils.color_propagation import colorize_scenes
    try:
        colorize_scenes(scene_refs_video,  before_path, colorized_scene_path, output_path)
    except (Exception, KeyboardInterrupt) as e:
        print(f"[ERROR] color propagation interrupted: {e}")
        if os.path.exists(output_path):
            print(f"[CLEANUP] Removing partial output video: {output_path}")
            os.remove(output_path)
        output_folder =  os.path.dirname(output_path) 
        if os.path.exists(output_folder) and os.path.isdir(output_folder):
            print(f"[CLEANUP] Removing output folder: {output_folder}")
            shutil.rmtree(output_folder)
    finally:
        import torch, gc
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
    return output_path


def colorize_scenes_prev_cached(scene_refs_video,  before_path, colorized_scene_path, output_path):
    """Propagating colors, using filecache to avoid reprocessing."""
    if os.path.exists(output_path):
        print(f"[CACHE] Propagated colored full video found: {output_path}")
        return output_path
    print("[INFO] Running Color propagation...")
    from Utils.color_propagation import colorize_scenes_prev
    try:
        colorize_scenes_prev(scene_refs_video,  before_path, colorized_scene_path, output_path)
    except (Exception, KeyboardInterrupt) as e:
        print(f"[ERROR] color propagation interrupted: {e}")
        if os.path.exists(output_path):
            print(f"[CLEANUP] Removing partial output video: {output_path}")
            os.remove(output_path)
        output_folder =  os.path.dirname(output_path) 
        if os.path.exists(output_folder) and os.path.isdir(output_folder):
            print(f"[CLEANUP] Removing output folder: {output_folder}")
            shutil.rmtree(output_folder)
    finally:
        import torch, gc
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
    return output_path

if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, "/opt/deepex")
    import sys
    sys.path.insert(0, "/workspace")
    parser = argparse.ArgumentParser(description="Propagate colors using reference frames and caching.")
    parser.add_argument("scene_refs_video", help="Path to the b&w extracted frames video")
    parser.add_argument("before_path", help="Path to the input file for scene_split")
    parser.add_argument("colorized_scene_path", help="Path to the colored extracted frames video")
    parser.add_argument("output_path", help="Output video path (used for cache folder logic)")

    args = parser.parse_args()

    for path_arg in [args.scene_refs_video, args.before_path, args.colorized_scene_path]:
        if not os.path.exists(path_arg):
            print(f"[ERROR] Path does not exist: {path_arg}")
            exit(1)

    try:
        final_output = colorize_scenes_cached(
            scene_refs_video=args.scene_refs_video,
            before_path=args.before_path,
            colorized_scene_path=args.colorized_scene_path,
            output_path=args.output_path
        )
        print(f"✔️ Color propagation output saved at: {final_output}")
    except Exception as e:
        print(f"[ERROR] Colorization failed: {e}")
        exit(1)
