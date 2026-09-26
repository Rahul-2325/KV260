"""
Vitis AI 2.5 Quantization Script for LCAM-YOLOX
Converts FP32 PyTorch model to INT8 for KV260 DPU deployment

Usage:
  python quantize.py --quant_mode calib    # Run calibration first
  python quantize.py --quant_mode test     # Then test/export
"""

import os
import sys
import argparse
from pathlib import Path

# Add YOLOX to path
sys.path.insert(0, '/workspace/YOLOX')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

# Vitis AI imports
from pytorch_nndct.apis import torch_quantizer

# ==========================================
# CONFIGURATION
# ==========================================
WORKSPACE = '/workspace'
CKPT_PATH = f'{WORKSPACE}/models/lcam_yolox_wildfire_v32_compat.pth'
CALIB_DIR = f'{WORKSPACE}/calibration'
OUTPUT_DIR = f'{WORKSPACE}/results/quantize_result'
INPUT_SIZE = 640  # 640x640
BATCH_SIZE = 1    # KV260 batch=1 for inference
NUM_CALIB_IMAGES = 200  # 200 is plenty for calibration

# ==========================================
# LCAM-YOLOX MODEL DEFINITION (wraps backbone)
# ==========================================
class LCAMYoloxBackbone(nn.Module):
    """
    Wrapper around the modified CSPDarknet backbone
    Used for quantization testing
    """
    def __init__(self):
        super().__init__()
        from yolox.models.darknet import CSPDarknet
        self.backbone = CSPDarknet(
            dep_mul=0.33,
            wid_mul=0.50,
            act='lrelu'
        )
    
    def forward(self, x):
        outputs = self.backbone(x)
        # Return as tuple (FPGA-friendly)
        return outputs['dark3'], outputs['dark4'], outputs['dark5']


# ==========================================
# CALIBRATION DATA LOADER
# ==========================================
class CalibrationDataset(Dataset):
    """Loads images for quantization calibration"""
    
    def __init__(self, img_dir, num_images=200, input_size=640):
        self.img_dir = img_dir
        self.input_size = input_size
        
        all_images = sorted([
            f for f in os.listdir(img_dir) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        
        # Limit to num_images
        self.image_files = all_images[:num_images]
        print(f"Calibration dataset: {len(self.image_files)} images")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Resize to (input_size, input_size) - same as training
        img = img.resize((self.input_size, self.input_size), Image.BILINEAR)
        
        # Convert to numpy
        img = np.array(img, dtype=np.float32)
        
        # HWC -> CHW (channels first)
        img = img.transpose(2, 0, 1)
        
        # Normalize to [0, 1]
        # Note: YOLOX expects [0, 255] range without /255
        # But for DPU we typically normalize
        # Adjust based on your training preprocessing
        # img = img / 255.0  # Uncomment if you trained with normalization
        
        return torch.from_numpy(img)


# ==========================================
# MAIN QUANTIZATION FUNCTION
# ==========================================
def quantize(args):
    print("=" * 70)
    print(f"LCAM-YOLOX QUANTIZATION - Mode: {args.quant_mode}")
    print("=" * 70)
    
    # ---- Setup ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # ---- Load model ----
    print("\n[1/5] Loading LCAM-YOLOX model...")
    model = LCAMYoloxBackbone()
    
    # Load trained weights
    print(f"[2/5] Loading weights from {CKPT_PATH}...")
    ckpt = torch.load(CKPT_PATH, map_location='cpu')
    
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    
    # Filter only backbone weights
    backbone_state = {}
    for k, v in state_dict.items():
        if k.startswith('backbone.backbone.'):
            new_key = 'backbone.' + k.replace('backbone.backbone.', '')
            backbone_state[new_key] = v
    
    missing, unexpected = model.load_state_dict(backbone_state, strict=False)
    print(f"  ✓ Loaded {len(backbone_state)} tensors")
    print(f"  Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    
    model = model.to(device)
    model.eval()
    
    # ---- Setup calibration data ----
    print("\n[3/5] Setting up calibration data loader...")
    calib_dataset = CalibrationDataset(
        CALIB_DIR, 
        num_images=NUM_CALIB_IMAGES,
        input_size=INPUT_SIZE
    )
    calib_loader = DataLoader(
        calib_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    # ---- Create quantizer ----
    print("\n[4/5] Creating Vitis AI quantizer...")
    
    # Dummy input for tracing
    dummy_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE).to(device)
    
    quantizer = torch_quantizer(
        quant_mode=args.quant_mode,  # 'calib' or 'test'
        module=model,
        input_args=(dummy_input,),
        output_dir=OUTPUT_DIR,
        device=device
    )
    
    quant_model = quantizer.quant_model
    quant_model.eval()
    
    # ---- Run forward passes for calibration ----
    print(f"\n[5/5] Running {args.quant_mode} pass...")
    print(f"  Processing {len(calib_loader)} batches...")
    
    with torch.no_grad():
        for i, batch in enumerate(calib_loader):
            batch = batch.to(device)
            _ = quant_model(batch)
            
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(calib_loader)}] Done")
    
    # ---- Export ----
    if args.quant_mode == 'calib':
        print("\n[*] Exporting quantization config...")
        quantizer.export_quant_config()
        print(f"  ✓ Config saved to: {OUTPUT_DIR}")
    
    elif args.quant_mode == 'test':
        print("\n[*] Exporting quantized model (xmodel)...")
        quantizer.export_xmodel(
            output_dir=OUTPUT_DIR,
            deploy_check=False
        )
        print(f"  ✓ xmodel saved to: {OUTPUT_DIR}")
    
    print("\n" + "=" * 70)
    print(f"✓ {args.quant_mode.upper()} COMPLETE")
    print("=" * 70)


# ==========================================
# ENTRY POINT
# ==========================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--quant_mode',
        choices=['calib', 'test'],
        required=True,
        help='calib: gather statistics; test: export xmodel'
    )
    args = parser.parse_args()
    
    quantize(args)
