import cv2
import os
import mediapipe as mp
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from basicsr.utils.img_util import img2tensor, tensor2img
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean
from PIL import Image
from torchvision.transforms.functional import normalize
import threading
import tqdm
from Utils.main_utils import slice_video, get_video_duration, concat_videos, repair_video_file
import sys
import os
import os
import sys

# def suppress_cpp_stderr():
#     """
#     Suppress C++ backend stderr (TensorFlow Lite, MediaPipe etc.)
#     without affecting Python's sys.stderr (e.g., tqdm).
#     """
#     devnull_fd = os.open(os.devnull, os.O_WRONLY)
#     os.dup2(devnull_fd, 2)  # 2 is the file descriptor for stderr

# suppress_cpp_stderr()  # If you want to suppress MediaPipe's GPU logs



class MP_FH_GFPGAN_Pipeline:
    """
    1) detect “good” faces with MediaPipe → box_overscan
    2) hand off each crop to FaceRestoreHelper + GFPGAN in one batch
    3) smooth‐blend each restored patch back with cv2.seamlessClone
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = None,
        min_conf: float = 0.5,
        eye_thresh: float = 8.0,
        overscan: float = 0.6,
        eye_overscan: float = 1.45,
        gfpgan_weight: float = 0.5,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.min_conf = min_conf
        self.eye_thresh = eye_thresh
        self.overscan = overscan
        self.eye_overscan = eye_overscan
        self.weight = gfpgan_weight

        # MediaPipe for initial “good” face detection
        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=self.min_conf,
        )

        # FaceRestoreHelper for exact 5-point alignment (uses RetinaFace under the hood)
        self.face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            save_ext="png",
            use_parse=False,
            device=self.device,
            model_rootpath="models",
        )

        # GFPGAN v1-clean model
        self.gfpgan = (
            GFPGANv1Clean(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=2,
                decoder_load_path=None,
                fix_decoder=False,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True,
            )
            .to(self.device)
            .eval()
        )
        ck = torch.load(checkpoint_path, map_location="cpu")
        key = "params_ema" if "params_ema" in ck else "params"
        self.gfpgan.load_state_dict(ck[key], strict=True)

    @torch.no_grad()
    def enhance_eyes(
        self, orig_frame: np.ndarray, rest_frame: np.ndarray
    ) -> np.ndarray:
        """
        Poisson-blend the full eye (including lids) from the original frame
        back into the GFPGAN output, with an overscan factor to expand the mask.
        """
        import cv2
        import mediapipe as mp
        import numpy as np

        out = rest_frame.copy()
        h, w = orig_frame.shape[:2]

        # 1) get face landmarks
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.1,
        )
        rgb = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        face_mesh.close()
        if not results.multi_face_landmarks:
            return out
        # 2) full-eye landmark sets (clockwise)
        left_eye_idx = [
            33,
            7,
            163,
            144,
            145,
            153,
            154,
            155,
            133,
            173,
            157,
            158,
            159,
            160,
            161,
            246,
        ]
        right_eye_idx = [
            362,
            382,
            381,
            380,
            374,
            373,
            390,
            249,
            263,
            466,
            388,
            387,
            386,
            385,
            384,
            398,
        ]
        eye_groups = [left_eye_idx, right_eye_idx]

        for lm in results.multi_face_landmarks:
            for eye_idx in eye_groups:
                # 3) make convex hull mask around the eye landmarks
                pts = np.array(
                    [
                        (int(lm.landmark[i].x * w), int(lm.landmark[i].y * h))
                        for i in eye_idx
                    ],
                    dtype=np.int32,
                )
                hull = cv2.convexHull(pts)
                mask = np.zeros((h, w), np.uint8)
                cv2.fillConvexPoly(mask, hull, 255)

                # 4) overscan (dilate) the mask by a fraction of its box
                x, y, wb, hb = cv2.boundingRect(hull)
                pad = int(max(wb, hb) * (self.eye_overscan - 1) / 2)
                if pad > 0:
                    kern = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (pad * 2 + 1, pad * 2 + 1)
                    )
                    mask = cv2.dilate(mask, kern, iterations=1)

                # 5) compute center for seamlessClone
                M = cv2.moments(mask)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
                    h, w = out.shape[:2]
                    cx = int(np.clip(cx, 0, w - 1))
                    cy = int(np.clip(cy, 0, h - 1))
                    center = (cx, cy)
                    
                    # Resize mask if needed
                    if mask.shape[:2] != orig_frame.shape[:2]:
                        mask = cv2.resize(mask, (orig_frame.shape[1], orig_frame.shape[0]))
                    
                    try:
                        out = cv2.seamlessClone(
                            src=orig_frame,
                            dst=out,
                            mask=mask,
                            p=center,
                            flags=cv2.NORMAL_CLONE,
                        )
                    except cv2.error as e:
                        print(f"[WARN] seamlessClone failed: {e} → skipping this patch.")


        return out


    @torch.no_grad()
    def enhance_frame(self, img_bgr: np.ndarray) -> np.ndarray:
        H, W = img_bgr.shape[:2]
        out_frame = img_bgr.copy()
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
        # ─ PASS 1: MediaPipe face detection + filtering
        rois = []
        for det in self.detector.process(rgb).detections or []:
            if det.score[0] < self.min_conf:
                continue
    
            r = det.location_data.relative_bounding_box
            x1 = int(max(0, r.xmin) * W)
            y1 = int(max(0, r.ymin) * H)
            w_box = int(r.width * W)
            h_box = int(r.height * H)
            if w_box < 32 or h_box < 32:
                continue
    
            # eye bridge check
            kps = det.location_data.relative_keypoints
            ex0, ey0 = kps[0].x * W, kps[0].y * H
            ex1, ey1 = kps[1].x * W, kps[1].y * H
            if np.hypot(ex1 - ex0, ey1 - ey0) < self.eye_thresh:
                continue
    
            # overscan padding
            pad_w = int(w_box * self.overscan)
            pad_h = int(h_box * self.overscan)
            x1p = max(0, x1 - pad_w)
            y1p = max(0, y1 - pad_h)
            x2p = min(W, x1 + w_box + pad_w)
            y2p = min(H, y1 + h_box + pad_h)
    
            rois.append((x1p, y1p, x2p, y2p))
    
        if not rois:
            return img_bgr
    
        crops512, invMs, boxes = [], [], []
    
        for x1p, y1p, x2p, y2p in rois:
            patch = out_frame[y1p:y2p, x1p:x2p]
            if patch.size == 0:
                continue
    
            patch512 = cv2.resize(patch, (512, 512), interpolation=cv2.INTER_LINEAR)
    
            fh = self.face_helper
            fh.clean_all()
            fh.read_image(patch512)
            fh.get_face_landmarks_5(eye_dist_threshold=self.eye_thresh)
            if not fh.det_faces:
                continue
    
            fh.align_warp_face()
            fh.get_inverse_affine()
    
            for face_crop, invM in zip(fh.cropped_faces, fh.inverse_affine_matrices):
                crops512.append(face_crop)
                invMs.append(invM)
                boxes.append((x1p, y1p, x2p, y2p))  # Still use original ROI for each
    
        if not crops512:
            return img_bgr
    
        # ─ PASS 2: GFPGAN batch inference
        ts = []
        for f512 in crops512:
            t = img2tensor(f512.astype(np.float32) / 255.0, bgr2rgb=True, float32=True)
            normalize(t, (0.5,) * 3, (0.5,) * 3, inplace=True)
            ts.append(t)
        batch = torch.stack(ts, 0).to(self.device)
    
        with torch.cuda.amp.autocast():
            outs = self.gfpgan(batch, return_rgb=False, weight=self.weight)
            if isinstance(outs, (list, tuple)):
                outs = outs[0]
    
        restored512_list = [
            tensor2img(outs[i].cpu(), rgb2bgr=True, min_max=(-1, 1)).astype(np.uint8)
            for i in range(len(crops512))
        ]
    
        # ─ PASS 3: Affine-blend each restored crop back
        for rf512, invM, (x1p, y1p, x2p, y2p) in zip(restored512_list, invMs, boxes):
            w_box, h_box = x2p - x1p, y2p - y1p
            if w_box < 8 or h_box < 8:
                continue
    
            patch512 = cv2.warpAffine(
                rf512, invM, (512, 512),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE
            )
    
            mask512 = np.ones((512, 512), np.uint8) * 255
            mask512 = cv2.warpAffine(mask512, invM, (512, 512),
                                     flags=cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=0)
            k512 = max(31, ((min(w_box, h_box) // 20) | 1))
            mask512 = cv2.GaussianBlur(mask512.astype(np.float32), (k512, k512), 0) / 255.0
            mask512 = mask512[..., None]
    
            patchROI = cv2.resize(patch512, (w_box, h_box), interpolation=cv2.INTER_LINEAR)
            maskROI = cv2.resize(mask512, (w_box, h_box), interpolation=cv2.INTER_LINEAR)
            maskROI = maskROI[..., None]
            dstROI = out_frame[y1p:y2p, x1p:x2p].astype(np.float32) / 255.0
            patchROI_f = patchROI.astype(np.float32) / 255.0
            compROI = maskROI * patchROI_f + (1.0 - maskROI) * dstROI
            face_patch = (compROI * 255.0).clip(0, 255).astype(np.uint8)
    
            orig_patch = img_bgr[y1p:y2p, x1p:x2p]
            enhanced_patch = self.enhance_eyes(orig_patch, face_patch)
    
            out_frame[y1p:y2p, x1p:x2p] = enhanced_patch
    
        return out_frame



_thread_locals = threading.local()

def _get_pipeline(checkpoint_path, device, min_conf, eye_thresh, overscan, eye_overscan, weight):
    """Create (once per thread) a pipeline instance."""
    p = getattr(_thread_locals, "pipeline", None)
    if p is None:
        p = MP_FH_GFPGAN_Pipeline(
            checkpoint_path=checkpoint_path,
            device=device,
            min_conf=min_conf,
            eye_thresh=eye_thresh,
            overscan=overscan,
            eye_overscan=eye_overscan,
            gfpgan_weight=weight,
        )
        _thread_locals.pipeline = p
    return p

def _worker_frame(args):
    """Unpack arguments, get thread-local pipeline, run enhance_frame."""
    frame, cfg = args
    pipeline = _get_pipeline(**cfg)
    # we assume frame is already a copy
    return pipeline.enhance_frame(frame)

def enhance_video_faces_parallel(
    input_path: str,
    output_path: str,
    checkpoint_path: str,
    device: str,
    min_conf: float,
    eye_thresh: float,
    overscan: float,
    eye_overscan: float,
    gfpgan_weight: float,
    num_workers: int | None = None,
    max_frames: int | None = None,
):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open {input_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is not None:
        total = min(total, max_frames)

    # output writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    cfg = dict(
        checkpoint_path=checkpoint_path,
        device=device,
        min_conf=min_conf,
        eye_thresh=eye_thresh,
        overscan=overscan,
        eye_overscan=eye_overscan,
        weight=gfpgan_weight,
    )
    max_workers = num_workers or os.cpu_count()
    pending: dict[concurrent.futures.Future, int] = {}
    results: dict[int, np.ndarray]         = {}
    next_to_write = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        print(f"[INFO] Enhancing {total} frames with up to {max_workers} workers…")
        try:
            from stqdm import stqdm as tqdm
            from tqdm import tqdm
        except ImportError:
            from tqdm import tqdm
        for idx in tqdm(range(total), desc="Submitting frames"):
            ret, frame = cap.read()
            if not ret:
                break

            # throttle: wait for one to finish if we have too many in flight
            if len(pending) >= max_workers:
                done_fut = next(as_completed(pending))
                done_idx = pending.pop(done_fut)
                results[done_idx] = done_fut.result()

                # write out as many in‐order frames as we can
                while next_to_write in results:
                    writer.write(results.pop(next_to_write))
                    next_to_write += 1

            # submit the next frame
            fut = executor.submit(_worker_frame, (frame.copy(), cfg))
            pending[fut] = idx

        # after submitting all, drain the rest
        for done_fut in as_completed(pending):
            done_idx = pending[done_fut]
            results[done_idx] = done_fut.result()

        # write any remaining in order
        while next_to_write in results:
            writer.write(results.pop(next_to_write))
            next_to_write += 1

    cap.release()
    writer.release()
    try:
        pipeline = _get_pipeline(**cfg)
        # delete model attributes
        del pipeline.gfpgan
        del pipeline.face_helper
        del pipeline.detector
        del pipeline
    except Exception:
        pass
    torch.cuda.empty_cache()
    repair_video_file(output_path)
    print(f"[INFO] Written enhanced video → {output_path}")
    
def upscale_faces(
    input_path: str,
    output_path: str
):
    """Main function to upscale faces in a video."""
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Run the face restoration
    enhance_video_faces_parallel(
        input_path=input_path,
        output_path=output_path,
        checkpoint_path="models/GFPGANv1_clean.pth",
        device="cuda",
        min_conf=0.3,
        eye_thresh=12.0,
        overscan=0.6,
        eye_overscan=1.45,
        gfpgan_weight=0.5,
        num_workers=8,
    )




def upscale_faces_cached(input_path, output_path ):
    """Upscaling Faces, using filecache to avoid reprocessing."""
    
    if os.path.exists(output_path):
        print(f"[CACHE] Faces upscaled video found: {output_path}")
        return output_path
    print("[INFO] Running Faces Upscaling...")
    from Utils.face_restore_utils import upscale_faces
    try:
        upscale_faces(input_path, output_path)
    except (Exception, KeyboardInterrupt) as e:
        print(f"[ERROR] Faces upscaling interrupted: {e}")
        if os.path.exists(output_path):
            print(f"[CLEANUP] Removing partial output video: {output_path}")
            os.remove(output_path)
        raise
    return output_path



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="face restoration...")
    parser.add_argument("input", help="Path to the input video file")
    parser.add_argument("output", help="Path to save the upscaled output")
    
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input file does not exist: {args.input}")
        sys.exit(1)

    try:
        upscale_faces_cached(args.input, args.output)
    except Exception as e:
        print(f"[ERROR] Faces Upscaling failed: {e}")
        sys.exit(1)
