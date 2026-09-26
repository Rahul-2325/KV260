"""
quantize_detector_v2.py
Quantize FULL detector (backbone + LCAM + FPN/PAN + YOLOXHead convs).
Built on the PROVEN preprocessing/calibration from quantize_neck.py.

KEY: decode_in_inference=False disables meshgrid (the Day-6 constant-output killer).
STEP 1 GOAL: just find out if the head SURVIVES quantization this time
(does output vary with input?) BEFORE investing in fast_finetune.

Usage:
  python scripts/quantize_detector_v2.py --quant_mode calib
  python scripts/quantize_detector_v2.py --quant_mode test
"""
import os, sys, argparse, re
sys.path.insert(0, '/workspace/YOLOX')
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from pytorch_nndct.apis import torch_quantizer

WORKSPACE = '/workspace'
CKPT_PATH = f'{WORKSPACE}/models/lcam_yolox_wildfire_v32_compat.pth'
CALIB_DIR = f'{WORKSPACE}/calibration'
OUTPUT_DIR = f'{WORKSPACE}/exports/detector_3p0_v4'
INPUT_SIZE = 640
BATCH_SIZE = 1
NUM_CALIB_IMAGES = 300


class LCAMYOLOXDetector(nn.Module):
    """Full detector: backbone+LCAM+neck+head, with meshgrid decode DISABLED."""
    def __init__(self):
        super().__init__()
        from yolox.exp import get_exp
        exp = get_exp(exp_file=None, exp_name='yolox-s')
        exp.num_classes = 2
        exp.act = 'lrelu'
        self.model = exp.get_model()
        self.model.eval()
        # CRITICAL: disable meshgrid-based decoding (Day-6 constant-output cause)
        self.model.head.decode_in_inference = False

    def forward(self, x):
        return self.model(x)


def yolox_preproc_bgr(img_bgr, size=640):
    """EXACT inference preprocessing (same as quantize_neck.py): BGR, AR resize, 114 pad, raw 0-255."""
    h0, w0 = img_bgr.shape[:2]
    r = min(size / h0, size / w0)
    nw, nh = int(w0 * r), int(h0 * r)
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded = np.full((size, size, 3), 114, dtype=np.uint8)
    padded[:nh, :nw] = resized
    x = padded.astype(np.float32).transpose(2, 0, 1)   # CHW, BGR, 0-255
    return x


class CalibrationDataset(Dataset):
    def __init__(self, img_dir, num_images=300, input_size=640):
        self.img_dir = img_dir
        self.input_size = input_size
        fire_list = '/workspace/calibration_fire_list.txt'
        if os.path.exists(fire_list):
            with open(fire_list) as fh:
                self.image_files = [l.strip() for l in fh if l.strip()][:num_images]
            print("Using FIRE-RICH calibration list")
        else:
            all_images = sorted([f for f in os.listdir(img_dir)
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            idx = np.linspace(0, len(all_images)-1, num_images).astype(int)
            self.image_files = [all_images[i] for i in idx]
        prefixes = {}
        for f in self.image_files:
            p = re.match(r'[A-Za-z]+', f)
            p = p.group() if p else '?'
            prefixes[p] = prefixes.get(p, 0) + 1
        print(f"Calibration: {len(self.image_files)} images, prefix mix: {prefixes}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        img_bgr = cv2.imread(img_path)
        x = yolox_preproc_bgr(img_bgr, self.input_size)
        return torch.from_numpy(x)


def constant_output_check(quant_model):
    """THE KEY TEST: feed 2 very different inputs, confirm outputs DIFFER.
    If outputs are identical, the head broke during quantization (Day-6 bug)."""
    print("\n" + "=" * 70)
    print("CONSTANT-OUTPUT SANITY CHECK (the Day-6 failure detector)")
    print("=" * 70)
    with torch.no_grad():
        a = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE)          # all black
        b = torch.full((1, 3, INPUT_SIZE, INPUT_SIZE), 127.0)  # all bright
        out_a = quant_model(a)
        out_b = quant_model(b)
        oa = out_a[0] if isinstance(out_a, (list, tuple)) else out_a
        ob = out_b[0] if isinstance(out_b, (list, tuple)) else out_b
        oa = oa.detach().cpu().numpy()
        ob = ob.detach().cpu().numpy()
        diff = np.abs(oa - ob).max()
        print(f"  Output shape: {oa.shape}")
        print(f"  Max abs difference between black vs bright input: {diff:.6f}")
        if diff < 1e-4:
            print("  >>> FAILED: outputs are CONSTANT. Head broke in quantization.")
            print("  >>> Fall back to neck-only + finetune. Head stays on CPU.")
        else:
            print("  >>> PASSED: output VARIES with input. Head survived quantization!")
            print("  >>> Safe to proceed to fast_finetune next.")
    print("=" * 70)


def quantize(args):
    print("=" * 70)
    print(f"FULL DETECTOR QUANTIZATION v2 - Mode: {args.quant_mode}")
    print("=" * 70)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cpu')

    print("\n[1/5] Building full detector (decode_in_inference=False)...")
    model = LCAMYOLOXDetector()
    total = sum(p.numel() for p in model.parameters())
    print(f"  Params: {total:,} ({total/1e6:.2f}M)")

    print(f"\n[2/5] Loading weights...")
    ckpt = torch.load(CKPT_PATH, map_location='cpu')
    state_dict = ckpt['model']
    missing, unexpected = model.model.load_state_dict(state_dict, strict=False)
    print(f"  Loaded. Missing {len(missing)} Unexpected {len(unexpected)}")
    model = model.to(device); model.eval()

    print("\n[3/5] Testing float forward...")
    with torch.no_grad():
        out = model(torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE))
        o = out[0] if isinstance(out, (list, tuple)) else out
        print(f"  Float output shape: {o.shape}")

    print("\n[4/5] Calibration data...")
    calib_dataset = CalibrationDataset(CALIB_DIR, NUM_CALIB_IMAGES, INPUT_SIZE)
    calib_loader = DataLoader(calib_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("\n[5/5] Quantizer...")
    dummy_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    quantizer = torch_quantizer(
        quant_mode=args.quant_mode, module=model,
        input_args=(dummy_input,), output_dir=OUTPUT_DIR, device=device)
    quant_model = quantizer.quant_model; quant_model.eval()

    print(f"\nRunning {args.quant_mode}...")
    failed = 0
    with torch.no_grad():
        for i, batch in enumerate(calib_loader):
            try:
                _ = quant_model(batch)
            except Exception as e:
                failed += 1
                if failed <= 3:
                    print(f"  Warn batch {i}: {str(e)[:80]}")
                continue
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(calib_loader)}] Done")
    if failed > 0:
        print(f"\n  Failed: {failed}/{len(calib_loader)}")

    if args.quant_mode == 'calib':
        quantizer.export_quant_config()
        print(f"\nConfig saved to {OUTPUT_DIR}")
    elif args.quant_mode == 'test':
        # Run the constant-output check BEFORE exporting
        constant_output_check(quant_model)
        quantizer.export_xmodel(output_dir=OUTPUT_DIR, deploy_check=False)
        print(f"\nxmodel saved to {OUTPUT_DIR}")

    print("\n" + "=" * 70)
    print(f"{args.quant_mode.upper()} COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quant_mode', choices=['calib', 'test'], required=True)
    args = parser.parse_args()
    quantize(args)
