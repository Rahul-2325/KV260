"""
Convert .pth from new PyTorch/numpy to compat format for Vitis AI 2.5
Handles numpy._core → numpy.core remapping
"""
import sys
import os

# CRITICAL: Add numpy._core → numpy.core compat shim BEFORE torch import
import numpy as np
import numpy.core
sys.modules['numpy._core'] = numpy.core
sys.modules['numpy._core.multiarray'] = numpy.core.multiarray
sys.modules['numpy._core.numeric'] = numpy.core.numeric
sys.modules['numpy._core._multiarray_umath'] = numpy.core._multiarray_umath

import torch

INPUT = '/home/punnam_rahul/wildfire_project/models/lcam_yolox_wildfire_v3_BEST.pth'
OUTPUT = '/home/punnam_rahul/wildfire_project/models/lcam_yolox_wildfire_v3_compat.pth'

print(f"Loading: {INPUT}")
print(f"Input size: {os.path.getsize(INPUT)/1e6:.1f} MB")

# Load with numpy shim active
ckpt = torch.load(INPUT, map_location='cpu', weights_only=False)

print(f"\nCheckpoint keys: {list(ckpt.keys())}")

# Get model state dict
if 'model' in ckpt:
    model_state = ckpt['model']
elif 'state_dict' in ckpt:
    model_state = ckpt['state_dict']
else:
    model_state = ckpt

print(f"Number of tensors: {len(model_state)}")

# Clone each tensor to remove any non-tensor references
clean_state = {}
for k, v in model_state.items():
    if isinstance(v, torch.Tensor):
        clean_state[k] = v.detach().clone().contiguous()

# Save in minimal format
torch.save({'model': clean_state}, OUTPUT)

print(f"\n✓ Saved compat version: {OUTPUT}")
print(f"Output size: {os.path.getsize(OUTPUT)/1e6:.1f} MB")

# Verify it loads (without shim now, since we cleaned it)
print("\nVerifying load...")
test = torch.load(OUTPUT, map_location='cpu', weights_only=False)
print(f"  ✓ Loaded successfully")
print(f"  ✓ Tensors: {len(test['model'])}")

# Check LCAM tensors
lcam_keys = [k for k in test['model'].keys() if 'lcam' in k]
print(f"  ✓ LCAM tensors: {len(lcam_keys)}")

print(f"\nSample LCAM keys:")
for k in lcam_keys[:6]:
    print(f"    {k}: {test['model'][k].shape}")

# Check for new v3.1 specific tensor
v31_specific = [k for k in test['model'].keys() if 'channel_reduce_avg' in k]
print(f"\nv3.1-specific tensors (channel_reduce_avg): {len(v31_specific)}")
for k in v31_specific:
    print(f"    {k}: {test['model'][k].shape}")
