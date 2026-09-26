"""
Builds lcam_v8_lightweight.ipynb -- a Kaggle-ready notebook that trains
LCAM-YOLOX with a LIGHTER BACKBONE, keeping the proven v5 LCAM attention
gate (Hardsigmoid + StagedGlobalAvgPool -- the exact module deployed on
the KV260 hardware today) unchanged.

WHY this and not a new attention module (see PLAN.md §0.2b): all four
literature candidates surveyed (YOLO-FireAD, EFA-YOLO, ESCFM-YOLO, YOLOFM)
get competitive fire/smoke detection accuracy at 1.4-1.9M parameters,
against our current 8.98M (YOLOX-s, width=0.5). DPU compute (~41-43 ms/
frame) is still the dominant cost in the pipelined-throughput ceiling, so
a lighter backbone attacks the actual bottleneck directly. This keeps the
already-hardware-verified LCAM HLS kernel unchanged -- only the backbone
WIDTH changes (0.5 -> 0.25), which changes channel counts (and therefore
retrains cleanly from COCO-pretrained weights, same as v7) but does not
change the LCAM module's own interface or its HLS kernel at all.

Run this script locally/WSL (`python3 build_v8_notebook.py`) to regenerate
lcam_v8_lightweight.ipynb, then:
    cd <this dir>
    python3 -m kaggle.cli kernels push -p .
(kernel-metadata.json is also written here, reusing the same
sayedgamal99/smoke-fire-detection-yolo dataset source as v7 -- no new
Kaggle dataset upload needed, everything else is git-cloned/downloaded
fresh inside the kernel, same as v7).

Only queue this AFTER lcam-v7-retrain finishes -- same Kaggle account,
same GPU quota, no benefit to contending for the slot.
"""
import json
import os

# ---------------------------------------------------------------------
# v5 LCAM module -- UNCHANGED from v7 / from the board-deployed model.
# Hardsigmoid + StagedGlobalAvgPool. Copied verbatim so this file has no
# hidden dependency on the WSL wildfire_project tree.
# ---------------------------------------------------------------------
LCAM_V5_SOURCE = r'''import torch
import torch.nn as nn


class StagedGlobalAvgPool(nn.Module):
    def __init__(self, stage1_kernel=4):
        super().__init__()
        self.stage1 = nn.AvgPool2d(kernel_size=stage1_kernel, stride=stage1_kernel, ceil_mode=True)
        self.stage2 = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        return self.stage2(self.stage1(x))


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.avg_pool = StagedGlobalAvgPool(stage1_kernel=4)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, reduced, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(reduced, channels, 1, bias=False)
        )
        self.gate = nn.Hardsigmoid()
    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_h = torch.max(x, dim=2, keepdim=True)[0]
        max_pool = torch.max(max_h, dim=3, keepdim=True)[0]
        max_out = self.mlp(max_pool)
        return x * self.gate(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, channels, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.channel_reduce_avg = nn.Conv2d(channels, 1, 1, bias=False)
        self.channel_reduce_max = nn.Conv2d(channels, 1, 1, bias=False)
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.gate = nn.Hardsigmoid()
    def forward(self, x):
        avg_out = self.channel_reduce_avg(x)
        max_out = self.channel_reduce_max(x)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        return x * self.gate(self.conv(spatial_input))


class LCAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(channels, spatial_kernel)
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
'''

# Ablation variant for FPS lever #2 (PLAN.md §0.1): SpatialAttention
# dropped, ChannelAttention only. NOT used by default -- see USE_SPATIAL
# below. Kept here so the ablation is one flag-flip away, not a rewrite.
LCAM_CHANNEL_ONLY_SOURCE = LCAM_V5_SOURCE.replace(
    "    def forward(self, x):\n        x = self.channel_attention(x)\n        x = self.spatial_attention(x)\n        return x",
    "    def forward(self, x):\n        x = self.channel_attention(x)\n        return x  # SpatialAttention dropped (FPS ablation, PLAN.md \\u00a70.1 item 2)"
)

# ---------------------------------------------------------------------
# Configuration for this run
# ---------------------------------------------------------------------
BACKBONE_WIDTH = 0.25   # was 0.5 in v5/v7 -- the actual experiment variable
USE_SPATIAL = True      # set False to also test the channel-only ablation
EXP_NAME = "lcam_yolox_wildfire_v8"

LCAM_SOURCE = LCAM_V5_SOURCE if USE_SPATIAL else LCAM_CHANNEL_ONLY_SOURCE


def code_cell(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.splitlines(keepends=True)}


cell0 = r"""
import os, sys, subprocess
print("="*60); print("STAGE 1/5: ENVIRONMENT SETUP"); print("="*60)
import torch
print(f"PyTorch: {torch.__version__}  CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    raise RuntimeError("GPU not available! Set Accelerator to GPU.")

DFIRE_PATH = '/kaggle/input/datasets/sayedgamal99/smoke-fire-detection-yolo/data'
if not os.path.exists(DFIRE_PATH):
    raise RuntimeError(f"Dataset missing at {DFIRE_PATH}")
for split in ['train', 'val', 'test']:
    d = f'{DFIRE_PATH}/{split}/images'
    print(f"  {split}: {len(os.listdir(d)) if os.path.exists(d) else 0} images")

for pkg in ['loguru', 'pycocotools', 'tabulate', 'thop', 'tqdm', 'ninja', 'opencv-python']:
    r = subprocess.run(['pip', 'install', '-q', pkg], capture_output=True, text=True)
    print(f"  {'ok' if r.returncode==0 else 'FAIL'} {pkg}")
print("STAGE 1 COMPLETE")
"""

cell1 = r"""
import os, sys, shutil
print("="*60); print("STAGE 2/5: YOLOX SETUP"); print("="*60)
YOLOX_DIR = '/kaggle/working/YOLOX'
if os.path.exists(YOLOX_DIR):
    shutil.rmtree(YOLOX_DIR)
os.chdir('/kaggle/working')
if os.system('git clone https://github.com/Megvii-BaseDetection/YOLOX.git --quiet') != 0:
    raise RuntimeError("Failed to clone YOLOX")
if YOLOX_DIR not in sys.path:
    sys.path.insert(0, YOLOX_DIR)
os.environ['PYTHONPATH'] = f'{YOLOX_DIR}:{os.environ.get("PYTHONPATH", "")}'
for m in list(sys.modules.keys()):
    if m.startswith('yolox'):
        del sys.modules[m]
print("STAGE 2 COMPLETE")
"""

cell2 = r"""
import os, shutil, sys
print("="*60); print("CLEANING UP PREVIOUS OUTPUTS"); print("="*60)
for p in ['/kaggle/working/YOLOX_outputs', '/kaggle/working/final_models']:
    if os.path.exists(p):
        shutil.rmtree(p)
        print(f"removed {p}")
for m in list(sys.modules.keys()):
    if m.startswith('yolox'):
        del sys.modules[m]
print("CLEANUP COMPLETE")
"""

cell3_template = r"""
import os
YOLOX_DIR = '/kaggle/working/YOLOX'
print("="*60); print("STAGE 3/5: LCAM INTEGRATION (v5 gate, __SPATIAL_NOTE__)"); print("="*60)

lcam_code = '''__LCAM_SOURCE__'''

with open(f'{YOLOX_DIR}/yolox/models/lcam.py', 'w') as f:
    f.write(lcam_code)
print("LCAM written")

darknet_path = f'{YOLOX_DIR}/yolox/models/darknet.py'
with open(darknet_path, 'r') as f:
    text = f.read()

if 'from .lcam import LCAM' not in text:
    text = text.replace(
        'from .network_blocks import',
        'from .lcam import LCAM\nfrom .network_blocks import',
        1
    )
text = text.replace(
    'self.stem = Focus(3, base_channels, ksize=3, act=act)',
    'self.stem = BaseConv(3, base_channels, 3, 2, act=act)'
)
csp_idx = text.find('class CSPDarknet')
before_csp = text[:csp_idx]
csp_block = text[csp_idx:]
if 'self.lcam2 = LCAM' not in csp_block:
    fwd_idx = csp_block.find('    def forward(self, x)')
    lcam_init = '''
        # LCAM modules
        self.lcam2 = LCAM(base_channels * 2)
        self.lcam3 = LCAM(base_channels * 4)
        self.lcam4 = LCAM(base_channels * 8)
        self.lcam5 = LCAM(base_channels * 16)

'''
    csp_block = csp_block[:fwd_idx] + lcam_init + csp_block[fwd_idx:]
    replacements = [
        ('        x = self.dark2(x)\n        outputs["dark2"] = x',
         '        x = self.dark2(x)\n        x = self.lcam2(x)\n        outputs["dark2"] = x'),
        ('        x = self.dark3(x)\n        outputs["dark3"] = x',
         '        x = self.dark3(x)\n        x = self.lcam3(x)\n        outputs["dark3"] = x'),
        ('        x = self.dark4(x)\n        outputs["dark4"] = x',
         '        x = self.dark4(x)\n        x = self.lcam4(x)\n        outputs["dark4"] = x'),
        ('        x = self.dark5(x)\n        outputs["dark5"] = x',
         '        x = self.dark5(x)\n        x = self.lcam5(x)\n        outputs["dark5"] = x'),
    ]
    for old, new in replacements:
        csp_block = csp_block.replace(old, new)
text = before_csp + csp_block
with open(darknet_path, 'w') as f:
    f.write(text)
print("darknet.py patched")

base_path = f'{YOLOX_DIR}/yolox/exp/yolox_base.py'
with open(base_path, 'r') as f:
    content = f.read()
content = content.replace('self.act = "silu"', 'self.act = "lrelu"')
with open(base_path, 'w') as f:
    f.write(content)
print("LeakyReLU set")

mlflow_path = f'{YOLOX_DIR}/yolox/utils/mlflow_logger.py'
if os.path.exists(mlflow_path):
    with open(mlflow_path, 'r') as f:
        c = f.read()
    if 'import importlib.metadata' in c and '# import importlib.metadata' not in c:
        c = c.replace('import importlib.metadata', '# import importlib.metadata')
        with open(mlflow_path, 'w') as f:
            f.write(c)
        print("mlflow disabled")

model_utils_path = f'{YOLOX_DIR}/yolox/utils/model_utils.py'
if os.path.exists(model_utils_path):
    with open(model_utils_path, 'r') as f:
        c = f.read()
    old = 'flops, params = profile(deepcopy(model), inputs=(img,), verbose=False)'
    if old in c and 'except Exception' not in c:
        new = '''try:
        flops, params = profile(deepcopy(model), inputs=(img,), verbose=False)
    except Exception as e:
        flops = 0
        params = sum(p.numel() for p in model.parameters())'''
        c = c.replace(old, new)
        with open(model_utils_path, 'w') as f:
            f.write(c)
        print("thop profiler wrapped")

import sys, torch
for m in list(sys.modules.keys()):
    if m.startswith('yolox'):
        del sys.modules[m]
from yolox.exp import get_exp
exp = get_exp(exp_file=None, exp_name='yolox-s')
exp.width = __WIDTH__
exp.num_classes = 2
exp.act = 'lrelu'
model = exp.get_model()
model.eval()
total = sum(p.numel() for p in model.parameters())
lcam_count = sum(1 for _, m in model.named_modules() if m.__class__.__name__ == 'LCAM')
print(f"width={exp.width}  Parameters: {total:,} ({total/1e6:.2f}M)   LCAM count: {lcam_count}/4")
assert lcam_count == 4
has_hardsig = any(isinstance(m, torch.nn.Hardsigmoid) for m in model.modules())
print(f"Has Hardsigmoid: {has_hardsig}")
assert has_hardsig
with torch.no_grad():
    out = model(torch.randn(1, 3, 640, 640))
print("Forward 640x640: OK")
print(f"STAGE 3 COMPLETE -- compare this param count against v7's 8.98M")
"""

cell3 = (cell3_template
         .replace("__LCAM_SOURCE__", LCAM_SOURCE.replace("\\", "\\\\").replace("'''", "\\'\\'\\'"))
         .replace("__WIDTH__", str(BACKBONE_WIDTH))
         .replace("__SPATIAL_NOTE__", "channel+spatial" if USE_SPATIAL else "channel-only ablation"))

cell4_template = r"""
import os, json
from PIL import Image
from pathlib import Path
from tqdm import tqdm
print("="*60); print("STAGE 4/5: DATASET + WEIGHTS + CONFIG"); print("="*60)

DFIRE_PATH = '/kaggle/input/datasets/sayedgamal99/smoke-fire-detection-yolo/data'
COCO_DIR = '/kaggle/working/coco_dataset'
os.makedirs(f'{COCO_DIR}/annotations', exist_ok=True)

def yolo_to_coco(images_dir, labels_dir, output_json, split):
    if os.path.exists(output_json):
        print(f"  {split} already converted, skipping")
        return
    coco = {'images': [], 'annotations': [],
            'categories': [{'id': 1, 'name': 'smoke', 'supercategory': 'fire_detection'},
                           {'id': 2, 'name': 'fire', 'supercategory': 'fire_detection'}]}
    ann_id = 1
    image_files = sorted(list(Path(images_dir).glob('*.jpg')) + list(Path(images_dir).glob('*.png')))
    for img_id, img_path in enumerate(tqdm(image_files, desc=split, leave=False), start=1):
        try:
            with Image.open(img_path) as img:
                width, height = img.size
        except Exception:
            continue
        coco['images'].append({'id': img_id, 'file_name': img_path.name, 'width': width, 'height': height})
        label_path = Path(labels_dir) / (img_path.stem + '.txt')
        if not label_path.exists():
            continue
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                try:
                    cls, xc, yc, w, h = int(float(parts[0])), *map(float, parts[1:])
                except ValueError:
                    continue
                abs_w, abs_h = w * width, h * height
                abs_x, abs_y = (xc * width) - (abs_w / 2), (yc * height) - (abs_h / 2)
                if abs_w <= 0 or abs_h <= 0:
                    continue
                coco['annotations'].append({'id': ann_id, 'image_id': img_id, 'category_id': cls + 1,
                                             'bbox': [abs_x, abs_y, abs_w, abs_h], 'area': abs_w * abs_h,
                                             'iscrowd': 0, 'segmentation': []})
                ann_id += 1
    with open(output_json, 'w') as f:
        json.dump(coco, f)
    print(f"  {split}: {len(coco['images'])} images, {len(coco['annotations'])} annotations")

yolo_to_coco(f'{DFIRE_PATH}/train/images', f'{DFIRE_PATH}/train/labels',
             f'{COCO_DIR}/annotations/instances_train.json', 'train')
yolo_to_coco(f'{DFIRE_PATH}/val/images', f'{DFIRE_PATH}/val/labels',
             f'{COCO_DIR}/annotations/instances_val.json', 'val')

for split, src_split in [('train2017', 'train'), ('val2017', 'val')]:
    dst = f'{COCO_DIR}/{split}'
    os.makedirs(dst, exist_ok=True)
    os.system(f'ln -sf {DFIRE_PATH}/{src_split}/images/* {dst}/ 2>/dev/null')
    print(f"  {split}: {len(os.listdir(dst))} files linked")

# NOTE: still downloading the width=0.5 COCO-pretrained yolox_s.pth --
# width=0.25 means most conv weights won't shape-match, so this only
# seeds the parts that DO match (stem etc.) via strict=False. A width-0.25
# COCO-pretrained checkpoint isn't published by Megvii; training this
# variant leans more on the 30-epoch schedule than on transfer learning.
# Worth flagging honestly in the paper if this run's accuracy lags v7's
# for that reason, rather than assuming the lighter backbone itself is worse.
PRETRAINED = '/kaggle/working/pretrained/yolox_s.pth'
os.makedirs(os.path.dirname(PRETRAINED), exist_ok=True)
if not os.path.exists(PRETRAINED):
    os.system(f'wget -q https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth -O {PRETRAINED}')
    print(f"Downloaded pretrained COCO weights ({os.path.getsize(PRETRAINED)/1e6:.1f} MB)")
else:
    print(f"Pretrained weights already exist ({os.path.getsize(PRETRAINED)/1e6:.1f} MB)")

YOLOX_DIR = '/kaggle/working/YOLOX'
config_path = f'{YOLOX_DIR}/exps/custom/__EXP_NAME__.py'
os.makedirs(os.path.dirname(config_path), exist_ok=True)

config = '''import os
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super().__init__()
        self.depth = 0.33
        self.width = __WIDTH__
        self.act = "lrelu"
        self.num_classes = 2

        self.data_dir = "/kaggle/working/coco_dataset"
        self.train_ann = "instances_train.json"
        self.val_ann = "instances_val.json"
        self.name = "train2017"

        self.max_epoch = 30
        self.warmup_epochs = 3
        self.no_aug_epochs = 10
        self.eval_interval = 10
        self.print_interval = 50

        self.basic_lr_per_img = 0.01 / 64.0
        self.momentum = 0.937
        self.weight_decay = 0.0005

        self.input_size = (640, 640)
        self.test_size = (640, 640)
        self.multiscale_range = 0

        self.mosaic_prob = 0.5
        self.mixup_prob = 0.5
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.degrees = 10.0
        self.translate = 0.1
        self.mosaic_scale = (0.5, 1.5)
        self.mixup_scale = (0.5, 1.5)
        self.shear = 2.0
        self.enable_mixup = True

        self.test_conf = 0.001
        self.nmsthre = 0.65

        self.data_num_workers = 4
        self.seed = 0

        self.output_dir = "/kaggle/working/YOLOX_outputs"
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
'''.replace("__WIDTH__", "__WIDTH_VAL__")
with open(config_path, 'w') as f:
    f.write(config)
print(f"Training config written: {config_path}")
print("STAGE 4 COMPLETE")
"""

cell4 = (cell4_template
         .replace("__EXP_NAME__", EXP_NAME)
         .replace("__WIDTH_VAL__", str(BACKBONE_WIDTH)))

cell5_template = r"""
import os, subprocess, time
print("="*60); print("STAGE 5/5: TRAINING"); print("="*60)

YOLOX_DIR = '/kaggle/working/YOLOX'
EXP_FILE = f'{YOLOX_DIR}/exps/custom/__EXP_NAME__.py'
PRETRAINED = '/kaggle/working/pretrained/yolox_s.pth'

os.chdir(YOLOX_DIR)
env = os.environ.copy()
env['PYTHONPATH'] = f'{YOLOX_DIR}:{env.get("PYTHONPATH", "")}'

cmd = ['python3', 'tools/train.py', '-f', EXP_FILE, '-d', '1', '-b', '32', '--fp16', '-c', PRETRAINED]
print("Starting training (width=__WIDTH__, LCAM __SPATIAL_NOTE__, 30 epochs)...\n")

start = time.time()
process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            universal_newlines=True, bufsize=1)
for line in iter(process.stdout.readline, ''):
    print(line, end='', flush=True)
elapsed = time.time() - start
print(f"\nFinished in {elapsed/60:.1f} minutes")
"""

cell5 = (cell5_template
         .replace("__EXP_NAME__", EXP_NAME)
         .replace("__WIDTH__", str(BACKBONE_WIDTH))
         .replace("__SPATIAL_NOTE__", "channel+spatial" if USE_SPATIAL else "channel-only"))

cell6_template = r"""
import os, shutil
print("="*60); print("SAVING OUTPUTS"); print("="*60)

src_dir = '/kaggle/working/YOLOX_outputs/__EXP_NAME__'
dst_dir = '/kaggle/working/final_models'
os.makedirs(dst_dir, exist_ok=True)

if os.path.exists(src_dir):
    for f in os.listdir(src_dir):
        if f.endswith('.pth'):
            name = '__EXP_NAME___BEST.pth' if f == 'best_ckpt.pth' else \
                   '__EXP_NAME___LATEST.pth' if f == 'latest_ckpt.pth' else \
                   f'__EXP_NAME___{f}'
            shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, name))
            print(f"  saved {name}")
    for name, path in [('lcam_v8.py', '/kaggle/working/YOLOX/yolox/models/lcam.py'),
                        ('__EXP_NAME___config.py', '/kaggle/working/YOLOX/exps/custom/__EXP_NAME__.py')]:
        if os.path.exists(path):
            shutil.copy2(path, os.path.join(dst_dir, name))
            print(f"  saved {name}")
    log_src = os.path.join(src_dir, 'train_log.txt')
    if os.path.exists(log_src):
        shutil.copy2(log_src, os.path.join(dst_dir, 'train_log.txt'))
        print("  saved train_log.txt")
else:
    print(f"WARNING: {src_dir} not found -- training may not have completed")

print("\nFiles in final_models/:")
for f in sorted(os.listdir(dst_dir)):
    print(f"  {f}: {os.path.getsize(os.path.join(dst_dir, f))/1e6:.1f} MB")
print("\n*** CLICK 'Save Version' NOW or these files will be deleted ***")
"""

cell6 = cell6_template.replace("__EXP_NAME__", EXP_NAME)

nb = {
    "cells": [code_cell(c) for c in [cell0, cell1, cell2, cell3, cell4, cell5, cell6]],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.10"}},
    "nbformat": 4, "nbformat_minor": 5,
}

if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(out_dir, "lcam_v8_lightweight.ipynb"), "w") as f:
        json.dump(nb, f, indent=1)
    kmeta = {
        "id": "punnamrahul/lcam-v8-lightweight",
        "title": "lcam_v8_lightweight",
        "code_file": "lcam_v8_lightweight.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
        "keywords": ["gpu"],
        "dataset_sources": ["sayedgamal99/smoke-fire-detection-yolo"],
        "kernel_sources": [], "competition_sources": [], "model_sources": [],
    }
    with open(os.path.join(out_dir, "kernel-metadata.json"), "w") as f:
        json.dump(kmeta, f, indent=2)
    print(f"Wrote lcam_v8_lightweight.ipynb + kernel-metadata.json to {out_dir}")
    print(f"Config: width={BACKBONE_WIDTH}, USE_SPATIAL={USE_SPATIAL}, exp_name={EXP_NAME}")
