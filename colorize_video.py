from __future__ import print_function

import os
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transform_lib
from PIL import Image
from tqdm import tqdm

import lib.TestTransforms as transforms
from models.ColorVidNet import ColorVidNet
from models.FrameColor import frame_colorization
from models.NonlocalNet import VGG19_pytorch, WarpNet
from utils.util import (batch_lab2rgb_transpose_mc, mkdir_if_not,
                        tensor_lab2rgb, uncenter_l)
from utils.util_distortion import CenterPad, Normalize, RGB2Lab, ToTensor

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)
# ---------------------------------------------------------------------------- #
# 🧠 Model Loading: only once
# ---------------------------------------------------------------------------- #
def load_colorization_models_(frame_propagate =False, image_size=(432, 768), device='cuda'):
    class Opt:
        pass
    opt = Opt()
    opt.frame_propagate = frame_propagate
    opt.image_size = image_size
    opt.gpu_ids = [0]
    opt.cuda = True

    cudnn.benchmark = True
    print("running on GPU", opt.gpu_ids)

    nonlocal_net = WarpNet(1)
    colornet = ColorVidNet(7)
    vggnet = VGG19_pytorch()
    vggnet.load_state_dict(torch.load("data/vgg19_conv.pth"))
    for param in vggnet.parameters():
        param.requires_grad = False

    nonlocal_test_path = os.path.join("checkpoints/", "video_moredata_l1/nonlocal_net_iter_76000.pth")
    color_test_path = os.path.join("checkpoints/", "video_moredata_l1/colornet_iter_76000.pth")
    print("succesfully load nonlocal model: ", nonlocal_test_path)
    print("succesfully load color model: ", color_test_path)
    nonlocal_net.load_state_dict(torch.load(nonlocal_test_path))
    colornet.load_state_dict(torch.load(color_test_path))

    nonlocal_net.eval()
    colornet.eval()
    vggnet.eval()
    nonlocal_net.cuda()
    colornet.cuda()
    vggnet.cuda()
    return opt, nonlocal_net, colornet, vggnet


def load_colorization_models(frame_propagate=False, image_size=(432, 768), device='cuda'):
#def load_colorization_models(frame_propagate=False, image_size=(768, 432), device='cuda'):
    class Opt:
        pass

    opt = Opt()
    opt.frame_propagate = frame_propagate
    image_size = (720,1280)
    opt.image_size = image_size
    opt.gpu_ids = [0]
    opt.cuda = True

    cudnn.benchmark = True
    # print("🚀 Running on GPU", opt.gpu_ids)

    # ✅ Get absolute path to the current script’s directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # print(base_dir)
    # ✅ Construct absolute paths to weights
    data_dir = os.path.join(base_dir, "data")
    ckpt_dir = os.path.join(base_dir, "checkpoints", "video_moredata_l1")

    # This assumes working dir is /workspace/app and data is mounted under it
    # APP_ROOT = "/workspace/app"
    # data_dir = os.path.join(APP_ROOT, "data")
    # ckpt_dir = os.path.join(APP_ROOT, "checkpoints", "video_moredata_l1")


    vgg_weights = os.path.join(data_dir, "vgg19_conv.pth")
    nonlocal_weights = os.path.join(ckpt_dir, "nonlocal_net_iter_76000.pth")
    color_weights = os.path.join(ckpt_dir, "colornet_iter_76000.pth")

    # ✅ Load models
    vggnet = VGG19_pytorch()
    vggnet.load_state_dict(torch.load(vgg_weights))
    for param in vggnet.parameters():
        param.requires_grad = False

    nonlocal_net = WarpNet(1)
    nonlocal_net.load_state_dict(torch.load(nonlocal_weights))

    colornet = ColorVidNet(7)
    colornet.load_state_dict(torch.load(color_weights))

    # ✅ Move models to device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    vggnet = vggnet.to(device).eval()
    nonlocal_net = nonlocal_net.to(device).eval()
    colornet = colornet.to(device).eval()

    # print("✅ All models loaded from:")
    # print("   VGG19        →", vgg_weights)
    # print("   NonlocalNet  →", nonlocal_weights)
    # print("   ColorVidNet  →", color_weights)

    return opt, nonlocal_net, colornet, vggnet

# ---------------------------------------------------------------------------- #
# 🎨 Colorization per Video
# ---------------------------------------------------------------------------- #
def colorize_video(opt, input_video_path, reference_file, output_video_path, nonlocal_net, colornet, vggnet):
    # parameters for wls filter
    wls_filter_on = True
    lambda_value = 500
    sigma_color = 4

    transform = transforms.Compose([
        CenterPad(opt.image_size),
        transform_lib.CenterCrop(opt.image_size),
        RGB2Lab(), ToTensor(), Normalize()
    ])

    cap = cv2.VideoCapture(input_video_path)
    assert cap.isOpened(), f"Failed to open video: {input_video_path}"
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    ##
    ##resize = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    resize_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    resize_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ref_name = input_video_path if opt.frame_propagate else reference_file
    #print("reference name:", ref_name)

    if opt.frame_propagate:
        ret, ref_frame_bgr = cap.read()
        if not ret:
            raise ValueError("Could not read the first frame as reference")
        ref_frame_rgb = cv2.cvtColor(ref_frame_bgr, cv2.COLOR_BGR2RGB)
        frame_ref = Image.fromarray(ref_frame_rgb)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    else:
        frame_ref = Image.open(ref_name)

# # 🔄 Dynamic image size based on reference
#     if opt.image_size is None:
#         h, w = frame_ref.height, frame_ref.width
#         h = (h // 8) * 8
#         w = (w // 8) * 8
#         opt.image_size = (h, w)
#         print("📐 Auto-detected image size:", opt.image_size)

#     transform = transforms.Compose([
#         CenterPad(opt.image_size),
#         transform_lib.Resize(opt.image_size),
#         RGB2Lab(), ToTensor(), Normalize()
#     ])

    IB_lab_large = transform(frame_ref).unsqueeze(0).cuda()
    IB_lab = torch.nn.functional.interpolate(IB_lab_large, scale_factor=0.5, mode="bilinear")
    IB_l = IB_lab[:, 0:1, :, :]
    IB_ab = IB_lab[:, 1:3, :, :]

    with torch.no_grad():
        I_reference_lab = IB_lab
        I_reference_l = I_reference_lab[:, 0:1, :, :]
        I_reference_ab = I_reference_lab[:, 1:3, :, :]
        I_reference_rgb = tensor_lab2rgb(torch.cat((uncenter_l(I_reference_l), I_reference_ab), dim=1))
        features_B = vggnet(I_reference_rgb, ["r12", "r22", "r32", "r42", "r52"], preprocess=True)

    out_writer = None
    I_last_lab_predict = None

    with tqdm(total=total_frames, desc="Colorizing Video") as pbar:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)

            IA_lab_large = transform(frame_pil).unsqueeze(0).cuda()
            IA_lab = torch.nn.functional.interpolate(IA_lab_large, scale_factor=0.5, mode="bilinear")
            IA_l = IA_lab[:, 0:1, :, :]
            IA_ab = IA_lab[:, 1:3, :, :]

            if I_last_lab_predict is None:
                I_last_lab_predict = IB_lab if opt.frame_propagate else torch.zeros_like(IA_lab).cuda()

            with torch.no_grad():
                I_current_lab = IA_lab
                I_current_ab_predict, _, _ = frame_colorization(
                    I_current_lab,
                    I_reference_lab,
                    I_last_lab_predict,
                    features_B,
                    vggnet,
                    nonlocal_net,
                    colornet,
                    feature_noise=0,
                    temperature=1e-10,
                )
                I_last_lab_predict = torch.cat((IA_l, I_current_ab_predict), dim=1)

            curr_bs_l = IA_lab_large[:, 0:1, :, :]
            curr_predict = (
                torch.nn.functional.interpolate(I_current_ab_predict.data.cpu(), scale_factor=2, mode="bilinear") * 1.25
            )

            if wls_filter_on:
                guide_image = uncenter_l(curr_bs_l) * 255 / 100
                wls_filter = cv2.ximgproc.createFastGlobalSmootherFilter(
                    guide_image[0, 0, :, :].cpu().numpy().astype(np.uint8), lambda_value, sigma_color
                )
                curr_predict_a = wls_filter.filter(curr_predict[0, 0, :, :].cpu().numpy())
                curr_predict_b = wls_filter.filter(curr_predict[0, 1, :, :].cpu().numpy())
                curr_predict_a = torch.from_numpy(curr_predict_a).unsqueeze(0).unsqueeze(0)
                curr_predict_b = torch.from_numpy(curr_predict_b).unsqueeze(0).unsqueeze(0)
                curr_predict_filter = torch.cat((curr_predict_a, curr_predict_b), dim=1)
                IA_predict_rgb = batch_lab2rgb_transpose_mc(curr_bs_l[:32], curr_predict_filter[:32, ...])
            else:
                IA_predict_rgb = batch_lab2rgb_transpose_mc(curr_bs_l[:32], curr_predict[:32, ...])

            img_np = IA_predict_rgb.astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            if out_writer is None:
                h, w = img_bgr.shape[:2]
                ext = os.path.splitext(output_video_path)[1].lower()
                fourcc = cv2.VideoWriter_fourcc(*('mp4v' if ext == '.mp4' else 'XVID'))
                out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (resize_width, resize_height))

            #out_writer.write(img_bgr)
            out_writer.write(cv2.resize(img_bgr, (resize_width, resize_height), interpolation=cv2.INTER_CUBIC))
            pbar.update(1)

    cap.release()
    out_writer.release()
    print("Saved colorized video:", output_video_path)

#######

def colorize_video(opt, input_video_path, reference_file, output_video_path, nonlocal_net, colornet, vggnet):
    # Parameters for WLS filter
    wls_filter_on = True
    lambda_value = 500
    sigma_color = 4

    lambda_value = 200
    sigma_color = 2
    #print(lambda_value,sigma_color)

    # Use Resize instead of CenterCrop to preserve full content
    transform = transforms.Compose([
        transform_lib.Resize(opt.image_size, interpolation=Image.BICUBIC),
        RGB2Lab(), ToTensor(), Normalize()
    ])

    cap = cv2.VideoCapture(input_video_path)
    assert cap.isOpened(), f"Failed to open video: {input_video_path}"
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    resize_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    resize_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ref_name = input_video_path if opt.frame_propagate else reference_file

    # Load reference frame
    if opt.frame_propagate:
        ret, ref_frame_bgr = cap.read()
        if not ret:
            raise ValueError("Could not read the first frame as reference")
        ref_frame_rgb = cv2.cvtColor(ref_frame_bgr, cv2.COLOR_BGR2RGB)
        frame_ref = Image.fromarray(ref_frame_rgb)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    else:
        frame_ref = Image.open(ref_name).convert("RGB")

    IB_lab_large = transform(frame_ref).unsqueeze(0).cuda()
    IB_lab = torch.nn.functional.interpolate(IB_lab_large, scale_factor=0.5, mode="bilinear")
    IB_l = IB_lab[:, 0:1, :, :]
    IB_ab = IB_lab[:, 1:3, :, :]

    with torch.no_grad():
        I_reference_lab = IB_lab
        I_reference_l = I_reference_lab[:, 0:1, :, :]
        I_reference_ab = I_reference_lab[:, 1:3, :, :]
        I_reference_rgb = tensor_lab2rgb(torch.cat((uncenter_l(I_reference_l), I_reference_ab), dim=1))
        features_B = vggnet(I_reference_rgb, ["r12", "r22", "r32", "r42", "r52"], preprocess=True)

    out_writer = None
    I_last_lab_predict = None

    #with tqdm(total=total_frames, desc="Colorizing Video") as pbar:
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        IA_lab_large = transform(frame_pil).unsqueeze(0).cuda()
        IA_lab = torch.nn.functional.interpolate(IA_lab_large, scale_factor=0.5, mode="bilinear")
        IA_l = IA_lab[:, 0:1, :, :]
        IA_ab = IA_lab[:, 1:3, :, :]

        if I_last_lab_predict is None:
            I_last_lab_predict = IB_lab if opt.frame_propagate else torch.zeros_like(IA_lab).cuda()

        with torch.no_grad():
            I_current_ab_predict, _, _ = frame_colorization(
                IA_lab,
                I_reference_lab,
                I_last_lab_predict,
                features_B,
                vggnet,
                nonlocal_net,
                colornet,
                feature_noise=0,
                temperature=1e-10,
            )
            I_last_lab_predict = torch.cat((IA_l, I_current_ab_predict), dim=1)

        curr_bs_l = IA_lab_large[:, 0:1, :, :]
        curr_predict = (
            torch.nn.functional.interpolate(I_current_ab_predict.data.cpu(), scale_factor=2, mode="bilinear") * 0.9
        )

        if wls_filter_on:
            guide_image = uncenter_l(curr_bs_l) * 255 / 100
            wls_filter = cv2.ximgproc.createFastGlobalSmootherFilter(
                guide_image[0, 0, :, :].cpu().numpy().astype(np.uint8), lambda_value, sigma_color
            )
            curr_predict_a = wls_filter.filter(curr_predict[0, 0, :, :].cpu().numpy())
            curr_predict_b = wls_filter.filter(curr_predict[0, 1, :, :].cpu().numpy())
            curr_predict_a = torch.from_numpy(curr_predict_a).unsqueeze(0).unsqueeze(0)
            curr_predict_b = torch.from_numpy(curr_predict_b).unsqueeze(0).unsqueeze(0)
            curr_predict_filter = torch.cat((curr_predict_a, curr_predict_b), dim=1)
            IA_predict_rgb = batch_lab2rgb_transpose_mc(curr_bs_l[:32], curr_predict_filter[:32, ...])
        else:
            IA_predict_rgb = batch_lab2rgb_transpose_mc(curr_bs_l[:32], curr_predict[:32, ...])

        img_np = IA_predict_rgb.astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        if out_writer is None:
            h, w = img_bgr.shape[:2]
            ext = os.path.splitext(output_video_path)[1].lower()
            fourcc = cv2.VideoWriter_fourcc(*('mp4v' if ext == '.mp4' else 'XVID'))
            out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (resize_width, resize_height))

        out_writer.write(cv2.resize(img_bgr, (resize_width, resize_height), interpolation=cv2.INTER_CUBIC))
            #pbar.update(1)

    cap.release()
    out_writer.release()
   # print("✅ Saved colorized video:", output_video_path)




import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

# --- Assumed helper functions ---
# RGB2Lab, ToTensor, Normalize
# uncenter_l, tensor_lab2rgb, batch_lab2rgb_transpose_mc
# frame_colorization

def tile_frame_lab(lab_tensor, tile_size=256, stride=128):
    B, C, H, W = lab_tensor.shape
    tiles = []
    positions = []

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y_end = min(y + tile_size, H)
            x_end = min(x + tile_size, W)
            crop = lab_tensor[:, :, y:y_end, x:x_end]

            # Resize crop to fixed 256x256 for model compatibility
            resized_tile = F.interpolate(crop, size=(tile_size, tile_size), mode='bilinear', align_corners=False)

            # Save original crop size for merging
            tiles.append(resized_tile)
            positions.append((y, x, y_end - y, x_end - x))
    return tiles, positions, H, W

def merge_tiles_lab(ab_tiles, positions, H, W):
    merged_ab = torch.zeros((2, H, W), dtype=ab_tiles[0].dtype, device=ab_tiles[0].device)
    magnitude_map = torch.zeros((H, W), dtype=ab_tiles[0].dtype, device=ab_tiles[0].device)

    for tile, (y, x, h_orig, w_orig) in zip(ab_tiles, positions):
        # Crop the tile back to original shape before writing
        a = tile[0, 0, :h_orig, :w_orig]
        b = tile[0, 1, :h_orig, :w_orig]
        mag = torch.sqrt(a ** 2 + b ** 2)

        current_mag = magnitude_map[y:y+h_orig, x:x+w_orig]
        mask = mag > current_mag

        merged_ab[0, y:y+h_orig, x:x+w_orig][mask] = a[mask]
        merged_ab[1, y:y+h_orig, x:x+w_orig][mask] = b[mask]
        magnitude_map[y:y+h_orig, x:x+w_orig][mask] = mag[mask]

    return merged_ab.unsqueeze(0)



def colorize_video_(opt, input_video_path, reference_file, output_video_path, nonlocal_net, colornet, vggnet):
    # Parameters for WLS filter
    wls_filter_on = True
    lambda_value = 500
    sigma_color = 4

    # Use Resize instead of CenterCrop to preserve full content
    transform = transforms.Compose([
        transform_lib.Resize(opt.image_size, interpolation=Image.BICUBIC),
        RGB2Lab(), ToTensor(), Normalize()
    ])

    cap = cv2.VideoCapture(input_video_path)
    assert cap.isOpened(), f"Failed to open video: {input_video_path}"
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    resize_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    resize_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ref_name = input_video_path if opt.frame_propagate else reference_file

    # Load reference frame
    if opt.frame_propagate:
        ret, ref_frame_bgr = cap.read()
        if not ret:
            raise ValueError("Could not read the first frame as reference")
        ref_frame_rgb = cv2.cvtColor(ref_frame_bgr, cv2.COLOR_BGR2RGB)
        frame_ref = Image.fromarray(ref_frame_rgb)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    else:
        frame_ref = Image.open(ref_name).convert("RGB")

    IB_lab_large = transform(frame_ref).unsqueeze(0).cuda()
    IB_lab = torch.nn.functional.interpolate(IB_lab_large, scale_factor=0.5, mode="bilinear")
    IB_l = IB_lab[:, 0:1, :, :]
    IB_ab = IB_lab[:, 1:3, :, :]

    with torch.no_grad():
        I_reference_lab = IB_lab
        I_reference_l = I_reference_lab[:, 0:1, :, :]
        I_reference_ab = I_reference_lab[:, 1:3, :, :]
        I_reference_rgb = tensor_lab2rgb(torch.cat((uncenter_l(I_reference_l), I_reference_ab), dim=1))
        features_B = vggnet(I_reference_rgb, ["r12", "r22", "r32", "r42", "r52"], preprocess=True)

    out_writer = None
    I_last_lab_predict = None

    with tqdm(total=total_frames, desc="Colorizing Video") as pbar:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)

            IA_lab_large = transform(frame_pil).unsqueeze(0).cuda()
            IA_lab = torch.nn.functional.interpolate(IA_lab_large, scale_factor=0.5, mode="bilinear")
            IA_l = IA_lab[:, 0:1, :, :]
            IA_ab = IA_lab[:, 1:3, :, :]

            if I_last_lab_predict is None:
                I_last_lab_predict = IB_lab if opt.frame_propagate else torch.zeros_like(IA_lab).cuda()

            with torch.no_grad():
                ab_tiles = []
                tiles, positions, H, W = tile_frame_lab(IA_lab, tile_size=256, stride=128)

                for idx, tile in enumerate(tiles):
                    try:
                        # Resize reference and history to match tile
                        ref_resized = F.interpolate(I_reference_lab, size=(256, 256), mode='bilinear', align_corners=False)
                        last_resized = F.interpolate(I_last_lab_predict, size=(256, 256), mode='bilinear', align_corners=False)
                
                        ab_pred, _, _ = frame_colorization(
                            tile,
                            ref_resized,
                            last_resized,
                            features_B,  # optionally recompute features for ref_resized
                            vggnet,
                            nonlocal_net,
                            colornet,
                            feature_noise=0,
                            temperature=1e-10,
                        )
                        ab_tiles.append(ab_pred)
                    except Exception as e:
                        print(f"❌ Error in tile {idx} at {positions[idx][:2]}: {e}")


                if len(ab_tiles) == 0:
                    raise RuntimeError("All tiles failed during frame_colorization.")

                I_current_ab_predict = merge_tiles_lab(ab_tiles, positions, H, W)
                I_last_lab_predict = torch.cat((IA_l, I_current_ab_predict), dim=1)

            curr_bs_l = IA_lab_large[:, 0:1, :, :]
            curr_predict = (
                torch.nn.functional.interpolate(I_current_ab_predict.data.cpu(), scale_factor=2, mode="bilinear") * 1.25
            )

            if wls_filter_on:
                guide_image = uncenter_l(curr_bs_l) * 255 / 100
                wls_filter = cv2.ximgproc.createFastGlobalSmootherFilter(
                    guide_image[0, 0, :, :].cpu().numpy().astype(np.uint8), lambda_value, sigma_color
                )
                curr_predict_a = wls_filter.filter(curr_predict[0, 0, :, :].cpu().numpy())
                curr_predict_b = wls_filter.filter(curr_predict[0, 1, :, :].cpu().numpy())
                curr_predict_a = torch.from_numpy(curr_predict_a).unsqueeze(0).unsqueeze(0)
                curr_predict_b = torch.from_numpy(curr_predict_b).unsqueeze(0).unsqueeze(0)
                curr_predict_filter = torch.cat((curr_predict_a, curr_predict_b), dim=1)
                IA_predict_rgb = batch_lab2rgb_transpose_mc(curr_bs_l[:32], curr_predict_filter[:32, ...])
            else:
                IA_predict_rgb = batch_lab2rgb_transpose_mc(curr_bs_l[:32], curr_predict[:32, ...])

            img_np = IA_predict_rgb.astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            if out_writer is None:
                h, w = img_bgr.shape[:2]
                ext = os.path.splitext(output_video_path)[1].lower()
                fourcc = cv2.VideoWriter_fourcc(*('mp4v' if ext == '.mp4' else 'XVID'))
                out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (resize_width, resize_height))

            out_writer.write(cv2.resize(img_bgr, (resize_width, resize_height), interpolation=cv2.INTER_CUBIC))
            pbar.update(1)

    cap.release()
    out_writer.release()
    #print("✅ Saved colorized video:", output_video_path)



# ---------------------------------------------------------------------------- #
# 🚀 Final Entry Point
# ---------------------------------------------------------------------------- #
def colorize_video_main(input_video_path, reference_file, output_video_path, opt, nonlocal_net, colornet, vggnet):
    colorize_video(
        opt=opt,
        input_video_path=input_video_path,
        reference_file=reference_file,
        output_video_path=output_video_path,
        nonlocal_net=nonlocal_net,
        colornet=colornet,
        vggnet=vggnet,
    )
