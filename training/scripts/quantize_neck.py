"""
Quantize YOLOPAFPN (CSPDarknet + LCAM + FPN/PAN)
FIXED: calibration preprocessing now matches inference (BGR, aspect-ratio
resize + 114 pad, raw 0-255), and calibration images are sampled ACROSS
all prefixes (AoF/WEB/PublicDataset) instead of first-200-alphabetical.
"""
import os, sys, argparse
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
OUTPUT_DIR = f'{WORKSPACE}/exports/neck'
INPUT_SIZE = 640
BATCH_SIZE = 1
NUM_CALIB_IMAGES = 300        # use more of the 500 available


class LCAMYOLOXNeck(nn.Module):
    def __init__(self):
        super().__init__()
        from yolox.models.yolo_pafpn import YOLOPAFPN
        self.backbone = YOLOPAFPN(depth=0.33, width=0.50, act='lrelu')
    def forward(self, x):
        return self.backbone(x)


def yolox_preproc_bgr(img_bgr, size=640):
    """EXACT inference preprocessing: BGR, aspect-ratio resize, 114 pad, raw 0-255."""
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
        # Load curated FIRE-RICH list (from select_fire_images.py)
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
        # report prefix coverage
        import re
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
        img_bgr = cv2.imread(img_path)              # BGR, matches inference
        x = yolox_preproc_bgr(img_bgr, self.input_size)
        return torch.from_numpy(x)


def quantize(args):
    print("=" * 70)
    print(f"YOLOPAFPN NECK QUANTIZATION (FIXED CALIB) - Mode: {args.quant_mode}")
    print("=" * 70)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cpu')

    print("\n[1/5] Building YOLOPAFPN...")
    model = LCAMYOLOXNeck()
    total = sum(p.numel() for p in model.parameters())
    print(f"  Params: {total:,} ({total/1e6:.2f}M)")

    print(f"\n[2/5] Loading weights...")
    ckpt = torch.load(CKPT_PATH, map_location='cpu')
    state_dict = ckpt['model']
    backbone_state = {k: v for k, v in state_dict.items() if k.startswith('backbone.')}
    missing, unexpected = model.load_state_dict(backbone_state, strict=False)
    print(f"  Loaded {len(backbone_state)} tensors  Missing {len(missing)} Unexpected {len(unexpected)}")
    model = model.to(device); model.eval()

    print("\n[3/5] Testing forward (expect 3 outputs)...")
    with torch.no_grad():
        out = model(torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE))
    for i, o in enumerate(out):
        print(f"  Output [{i}]: {o.shape}")

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
