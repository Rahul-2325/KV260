"""
fire_baseline_check.py
Run the quantized full-detector on a KNOWN fire+smoke image and report
the actual fire/smoke combined scores + the obj/class ALIGNMENT diagnostic.
This establishes the BASELINE before any fast_finetune.

Run inside Docker:
  python scripts/fire_baseline_check.py
"""
import os, sys
sys.path.insert(0, '/workspace/YOLOX')
import torch
import torch.nn as nn
import numpy as np
import cv2
from pytorch_nndct.apis import torch_quantizer

WORKSPACE = '/workspace'
CKPT_PATH = f'{WORKSPACE}/models/lcam_yolox_wildfire_v32_compat.pth'
OUTPUT_DIR = f'{WORKSPACE}/exports/detector_v2'
TEST_IMG = f'{WORKSPACE}/dataset/data/test/images/WEB09971.jpg'
INPUT_SIZE = 640
STRIDES = [8, 16, 32]   # YOLOX strides for the 3 feature levels


class LCAMYOLOXDetector(nn.Module):
    def __init__(self):
        super().__init__()
        from yolox.exp import get_exp
        exp = get_exp(exp_file=None, exp_name='yolox-s')
        exp.num_classes = 2
        exp.act = 'lrelu'
        self.model = exp.get_model()
        self.model.eval()
        self.model.head.decode_in_inference = False
    def forward(self, x):
        return self.model(x)


def yolox_preproc_bgr(img_bgr, size=640):
    h0, w0 = img_bgr.shape[:2]
    r = min(size / h0, size / w0)
    nw, nh = int(w0 * r), int(h0 * r)
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded = np.full((size, size, 3), 114, dtype=np.uint8)
    padded[:nh, :nw] = resized
    x = padded.astype(np.float32).transpose(2, 0, 1)
    return x, r


def main():
    print("=" * 70)
    print("FIRE BASELINE CHECK (quantized detector_v2, NO finetune yet)")
    print(f"Image: {os.path.basename(TEST_IMG)} (ground truth: smoke + fire)")
    print("=" * 70)

    device = torch.device('cpu')

    # Build + load
    model = LCAMYOLOXDetector()
    ckpt = torch.load(CKPT_PATH, map_location='cpu')
    model.model.load_state_dict(ckpt['model'], strict=False)
    model = model.to(device); model.eval()

    # Reload as QUANTIZED model (test mode reconstructs INT8-simulated model)
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    quantizer = torch_quantizer(
        quant_mode='test', module=model,
        input_args=(dummy,), output_dir=OUTPUT_DIR, device=device)
    qmodel = quantizer.quant_model; qmodel.eval()

    # Preprocess the fire image
    img = cv2.imread(TEST_IMG)
    assert img is not None, f"Could not read {TEST_IMG}"
    x, ratio = yolox_preproc_bgr(img, INPUT_SIZE)
    xt = torch.from_numpy(x).unsqueeze(0)

    # Forward (raw output, decode_in_inference=False)
    with torch.no_grad():
        out = qmodel(xt)
    raw = out[0] if isinstance(out, (list, tuple)) else out
    raw = raw.detach().cpu().numpy()[0]   # (8400, 7)
    print(f"\nRaw output shape: {raw.shape}  (cx,cy,w,h,obj,smoke,fire)")

    # Split
    obj_logit   = raw[:, 4]
    smoke_logit = raw[:, 5]
    fire_logit  = raw[:, 6]

    def sig(z): return 1.0 / (1.0 + np.exp(-z))
    obj   = sig(obj_logit)
    smoke = sig(smoke_logit)
    fire  = sig(fire_logit)

    comb_smoke = obj * smoke
    comb_fire  = obj * fire

    print("\n--- CLASS-ALONE scores (sigmoid of class logit) ---")
    print(f"  smoke: max={smoke.max():.3f}  mean={smoke.mean():.3f}  cells>0.5: {(smoke>0.5).sum()}")
    print(f"  fire : max={fire.max():.3f}  mean={fire.mean():.3f}  cells>0.5: {(fire>0.5).sum()}")

    print("\n--- OBJECTNESS ---")
    print(f"  obj  : max={obj.max():.3f}  mean={obj.mean():.3f}  cells>0.5: {(obj>0.5).sum()}")

    print("\n--- COMBINED scores (obj * class) = WHAT ACTUALLY MATTERS ---")
    print(f"  smoke: max={comb_smoke.max():.3f}  cells>0.3: {(comb_smoke>0.3).sum()}  cells>0.1: {(comb_smoke>0.1).sum()}")
    print(f"  fire : max={comb_fire.max():.3f}  cells>0.3: {(comb_fire>0.3).sum()}  cells>0.1: {(comb_fire>0.1).sum()}")

    # ALIGNMENT DIAGNOSTIC: do obj and fire peak in the SAME cells?
    print("\n--- ALIGNMENT DIAGNOSTIC (the suspected root cause) ---")
    top_fire_cell = np.argmax(fire)
    top_obj_cell  = np.argmax(obj)
    print(f"  Cell with highest FIRE class score: idx={top_fire_cell}")
    print(f"    at that cell:  obj={obj[top_fire_cell]:.3f}  fire={fire[top_fire_cell]:.3f}  combined={comb_fire[top_fire_cell]:.3f}")
    print(f"  Cell with highest OBJECTNESS: idx={top_obj_cell}")
    print(f"    at that cell:  obj={obj[top_obj_cell]:.3f}  fire={fire[top_obj_cell]:.3f}  combined={comb_fire[top_obj_cell]:.3f}")
    if top_fire_cell != top_obj_cell:
        print("  >>> MISALIGNED: fire-class and objectness peak in DIFFERENT cells.")
        print("  >>> This is the collapse mechanism — combined fire score stays low.")
    else:
        print("  >>> ALIGNED: fire and objectness peak together. Good sign.")

    print("\n" + "=" * 70)
    print("VERDICT")
    if comb_fire.max() > 0.3:
        print("  FIRE detectable (combined > 0.3). May NOT need finetune for fire.")
    elif comb_fire.max() > 0.1:
        print("  FIRE weak (combined 0.1-0.3). Borderline; finetune likely helps.")
    else:
        print("  FIRE COLLAPSED (combined < 0.1). This is the problem; finetune is the lever.")
    print(f"  (smoke combined max = {comb_smoke.max():.3f} for comparison)")
    print("=" * 70)


if __name__ == '__main__':
    main()
