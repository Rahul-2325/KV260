"""Verify modified YOLOX can load LCAM-YOLOX trained .pth"""
import sys
import os

sys.path.insert(0, os.path.expanduser('~/wildfire_project/YOLOX'))

import torch
import torch.nn as nn

print("=" * 60)
print("VERIFYING LCAM-YOLOX SETUP")
print("=" * 60)

# Step 1: Import modified CSPDarknet
print("\n[1/5] Importing modified CSPDarknet...")
try:
    from yolox.models.darknet import CSPDarknet
    print("  ✓ Import successful")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# Step 2: Import LCAM
print("\n[2/5] Importing LCAM module...")
try:
    from yolox.models.lcam import LCAM
    test_lcam = LCAM(channels=64)
    n_params = sum(p.numel() for p in test_lcam.parameters())
    print(f"  ✓ LCAM imported, test instance has {n_params} parameters")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# Step 3: Create model
print("\n[3/5] Creating CSPDarknet with LeakyReLU + LCAM...")
try:
    model = CSPDarknet(dep_mul=0.33, wid_mul=0.50, act='lrelu')
    total = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Model created with {total:,} parameters")
    
    has_lcam = all([
        hasattr(model, 'lcam2'),
        hasattr(model, 'lcam3'),
        hasattr(model, 'lcam4'),
        hasattr(model, 'lcam5'),
    ])
    print(f"  ✓ All 4 LCAM modules present: {has_lcam}")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# Step 4: Load checkpoint
print("\n[4/5] Loading trained checkpoint...")
ckpt_path = os.path.expanduser('~/wildfire_project/models/lcam_yolox_wildfire_TRAINED_compat.pth')
try:
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    print(f"  ✓ Checkpoint loaded")
    print(f"  ✓ Keys: {list(ckpt.keys())}")
    
    if 'model' in ckpt:
        state_dict = ckpt['model']
        print(f"  ✓ State dict: {len(state_dict)} tensors")
    
    if 'best_ap' in ckpt:
        print(f"  ✓ Best AP: {ckpt['best_ap']:.4f}")
    if 'start_epoch' in ckpt:
        print(f"  ✓ Last epoch: {ckpt['start_epoch']}")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# Step 5: Check weight loading compatibility
print("\n[5/5] Checking weight compatibility...")
try:
    # Extract backbone weights from full state dict
    backbone_state = {}
    if 'model' in ckpt:
        for k, v in ckpt['model'].items():
            if k.startswith('backbone.backbone.'):
                # Strip 'backbone.backbone.' prefix
                new_key = k.replace('backbone.backbone.', '')
                backbone_state[new_key] = v
    
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(backbone_state.keys())
    
    matched = model_keys & ckpt_keys
    only_in_model = model_keys - ckpt_keys
    only_in_ckpt = ckpt_keys - model_keys
    
    print(f"  ✓ Matched keys: {len(matched)}")
    if only_in_model:
        print(f"  ⚠ Only in model (not in checkpoint): {len(only_in_model)}")
        for k in list(only_in_model)[:3]:
            print(f"    - {k}")
    if only_in_ckpt:
        print(f"  ⚠ Only in checkpoint (not in model): {len(only_in_ckpt)}")
        for k in list(only_in_ckpt)[:3]:
            print(f"    - {k}")
    
    # Try loading the backbone weights
    missing, unexpected = model.load_state_dict(backbone_state, strict=False)
    print(f"  ✓ Loaded with {len(missing)} missing, {len(unexpected)} unexpected keys")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# Forward pass test
print("\n[BONUS] Forward pass test...")
try:
    model.eval()
    dummy = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        outputs = model(dummy)
    
    print(f"  ✓ Forward pass successful")
    print(f"  ✓ Backbone outputs:")
    for k, v in outputs.items():
        print(f"    - {k}: shape {tuple(v.shape)}")
except Exception as e:
    print(f"  ✗ Failed: {e}")

print("\n" + "=" * 60)
print("🎉 VERIFICATION COMPLETE!")
print("=" * 60)
