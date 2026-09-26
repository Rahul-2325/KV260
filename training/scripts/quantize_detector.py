import os
import sys
import argparse

sys.path.insert(0, '/workspace/YOLOX')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

from pytorch_nndct.apis import torch_quantizer

WORKSPACE = '/workspace'
CKPT_PATH = f'{WORKSPACE}/models/lcam_yolox_wildfire_v32_compat.pth'
CALIB_DIR = f'{WORKSPACE}/calibration'
OUTPUT_DIR = f'{WORKSPACE}/exports/detector'
INPUT_SIZE = 640
BATCH_SIZE = 1
NUM_CALIB_IMAGES = 200


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


class CalibrationDataset(Dataset):
    def __init__(self, img_dir, num_images=200, input_size=640):
        self.img_dir = img_dir
        self.input_size = input_size
        all_images = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        self.image_files = all_images[:num_images]
        print(f"Calibration dataset: {len(self.image_files)} images")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.input_size, self.input_size), Image.BILINEAR)
        img = np.array(img, dtype=np.float32)
        img = img.transpose(2, 0, 1)
        return torch.from_numpy(img)


def quantize(args):
    print("=" * 70)
    print(f"DETECTOR QUANTIZATION (decode disabled) - Mode: {args.quant_mode}")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cpu')

    print("\n[1/5] Building model with decode_in_inference=False...")
    model = LCAMYOLOXDetector()

    print(f"\n[2/5] Loading weights...")
    ckpt = torch.load(CKPT_PATH, map_location='cpu')
    state_dict = ckpt['model']
    missing, unexpected = model.model.load_state_dict(state_dict, strict=False)
    print(f"  Loaded {len(state_dict)} tensors, Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    model = model.to(device)
    model.eval()

    print("\n[3/5] Testing forward...")
    with torch.no_grad():
        test_in = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
        test_out = model(test_in)
    print(f"  Output shape: {test_out.shape}")

    print("\n[4/5] Calibration data...")
    calib_dataset = CalibrationDataset(CALIB_DIR, num_images=NUM_CALIB_IMAGES, input_size=INPUT_SIZE)
    calib_loader = DataLoader(calib_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print("\n[5/5] Quantizer...")
    dummy_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)

    quantizer = torch_quantizer(
        quant_mode=args.quant_mode,
        module=model,
        input_args=(dummy_input,),
        output_dir=OUTPUT_DIR,
        device=device
    )

    quant_model = quantizer.quant_model
    quant_model.eval()

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
        print(f"\n  Total failed: {failed}/{len(calib_loader)}")

    if args.quant_mode == 'calib':
        quantizer.export_quant_config()
        print(f"\n✓ Config saved to {OUTPUT_DIR}")
    elif args.quant_mode == 'test':
        quantizer.export_xmodel(output_dir=OUTPUT_DIR, deploy_check=False)
        print(f"\n✓ xmodel saved to {OUTPUT_DIR}")

    print("\n" + "=" * 70)
    print(f"✓ {args.quant_mode.upper()} COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quant_mode', choices=['calib', 'test'], required=True)
    args = parser.parse_args()
    quantize(args)
