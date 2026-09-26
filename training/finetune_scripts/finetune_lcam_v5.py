"""
LCAM-YOLOX v5 Fine-Tuning Script (50 epochs)

Fine-tunes the v5 architecture (hardsigmoid + staged global pooling) starting
from the v3.2 weights. v5 has NO new learnable parameters (hardsigmoid and
staged avgpool are parameter-free), so every conv weight transfers exactly and
we only adapt to the changed activation/pooling behaviour.

Usage on Kaggle:
  - Upload this folder (with the v5 YOLOX/yolox/models/lcam.py) as a dataset
  - Attach the D-Fire dataset (smoke-fire-detection-yolo)
  - Run in a Kaggle notebook with GPU T4 enabled
"""

import os, sys, time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, '/kaggle/working/YOLOX')

print("=" * 70)
print("LCAM-YOLOX v5 Fine-Tuning (hardsigmoid + staged pool)")
print("=" * 70)

CONFIG = {
    'dataset_root':    '/kaggle/input/smoke-fire-detection-yolo/data',
    # INPUT: your v3.2 weights (starting point)
    'checkpoint_path': '/kaggle/working/lcam_yolox_wildfire_v32_compat.pth',
    'output_dir':      '/kaggle/working/output_v5',
    'depth': 0.33, 'width': 0.5, 'num_classes': 2, 'input_size': 640,
    'epochs': 50,             # <-- 50 epochs
    'batch_size': 16,
    'learning_rate': 1e-4,    # low LR for fine-tuning
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'num_workers': 4,
    'print_freq': 50,
    'save_freq': 5,
}

print("\nConfiguration:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")


class DFireDataset(Dataset):
    """D-Fire YOLO format: images/IMG.jpg, labels/IMG.txt (cls x y w h normalized)."""
    def __init__(self, root, split='train', input_size=640, augment=True):
        self.root = Path(root)
        self.split = split
        self.input_size = input_size
        self.augment = augment and (split == 'train')
        self.img_dir = self.root / split / 'images'
        self.lbl_dir = self.root / split / 'labels'
        self.images = sorted(list(self.img_dir.glob('*.jpg')))
        print(f"  {split}: {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def load_labels(self, img_path):
        lbl_path = self.lbl_dir / (img_path.stem + '.txt')
        if not lbl_path.exists():
            return np.zeros((0, 5), dtype=np.float32)
        labels = []
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls = int(parts[0])
                    x, y, w, h = map(float, parts[1:])
                    labels.append([cls, x, y, w, h])
        return np.array(labels, dtype=np.float32) if labels else np.zeros((0, 5), dtype=np.float32)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Cannot read: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        labels = self.load_labels(img_path)
        img = cv2.resize(img, (self.input_size, self.input_size))
        if self.augment:
            if np.random.random() > 0.5:
                img = img[:, ::-1, :].copy()
                if len(labels) > 0:
                    labels[:, 1] = 1 - labels[:, 1]
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        img = torch.from_numpy(img)
        return img, labels


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    # ====== Build v5 model ======
    print("\n[1/4] Building LCAM-YOLOX v5 model...")
    from yolox.exp import get_exp
    exp = get_exp(exp_file=None, exp_name='yolox_s')
    exp.depth = CONFIG['depth']
    exp.width = CONFIG['width']
    exp.num_classes = CONFIG['num_classes']
    exp.input_size = (CONFIG['input_size'], CONFIG['input_size'])
    exp.act = 'lrelu'
    model = exp.get_model()
    print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")

    # Sanity: confirm v5 architecture is actually loaded
    from yolox.models.lcam import LCAM, ChannelAttention
    import inspect
    ca_src = inspect.getsource(ChannelAttention)
    has_hardsig = 'Hardsigmoid' in ca_src
    has_staged = 'StagedGlobalAvgPool' in inspect.getsource(sys.modules['yolox.models.lcam'])
    print(f"  v5 check -> Hardsigmoid: {has_hardsig}, StagedGlobalAvgPool: {has_staged}")
    if not (has_hardsig and has_staged):
        print("  WARNING: v5 architecture NOT detected! Check lcam.py is the v5 version.")

    # ====== Load v3.2 weights ======
    print("\n[2/4] Loading v3.2 checkpoint for fine-tuning...")
    ckpt = torch.load(CONFIG['checkpoint_path'], map_location='cpu')
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  Loaded (strict=False). Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    if missing:
        print("  Missing keys (random init - should only be non-weight buffers, if any):")
        for k in missing[:8]:
            print(f"    - {k}")
    model = model.to(device)

    # ====== Data ======
    print("\n[3/4] Setting up data...")
    train_ds = DFireDataset(CONFIG['dataset_root'], split='train',
                            input_size=CONFIG['input_size'], augment=True)
    val_ds = DFireDataset(CONFIG['dataset_root'], split='valid',
                          input_size=CONFIG['input_size'], augment=False)

    def collate_fn(batch):
        imgs, lbls = zip(*batch)
        imgs = torch.stack(imgs)
        max_n = max(max(len(l) for l in lbls), 1)
        padded = np.zeros((len(lbls), max_n, 5), dtype=np.float32)
        for i, l in enumerate(lbls):
            if len(l) > 0:
                padded[i, :len(l)] = l
        return imgs, torch.from_numpy(padded)

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True,
                              num_workers=CONFIG['num_workers'], collate_fn=collate_fn,
                              pin_memory=True)
    print(f"  Train batches: {len(train_loader)}")

    # ====== Optimizer ======
    optimizer = optim.SGD(model.parameters(), lr=CONFIG['learning_rate'],
                          momentum=CONFIG['momentum'], weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])

    # ====== Train ======
    print("\n[4/4] Starting fine-tuning (50 epochs)...")
    best_loss = float('inf')
    for epoch in range(CONFIG['epochs']):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()
        for i, (imgs, targets) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(imgs, targets)
            loss = outputs['total_loss']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if (i + 1) % CONFIG['print_freq'] == 0:
                avg = epoch_loss / (i + 1)
                lr = optimizer.param_groups[0]['lr']
                elapsed = time.time() - epoch_start
                fps = (i + 1) * CONFIG['batch_size'] / elapsed
                print(f"  Epoch {epoch+1}/{CONFIG['epochs']} [{i+1}/{len(train_loader)}] "
                      f"loss={avg:.4f} lr={lr:.6f} fps={fps:.1f}")
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} complete: avg loss={avg_loss:.4f} "
              f"({time.time()-epoch_start:.1f}s)")

        if (epoch + 1) % CONFIG['save_freq'] == 0 or (epoch + 1) == CONFIG['epochs']:
            ckpt_path = os.path.join(CONFIG['output_dir'], f'v5_epoch_{epoch+1}.pth')
            torch.save({'model': model.state_dict(), 'epoch': epoch + 1, 'loss': avg_loss}, ckpt_path)
            print(f"  Saved: {ckpt_path}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(CONFIG['output_dir'], 'lcam_yolox_v5_finetuned_best.pth')
            torch.save({'model': model.state_dict(), 'epoch': epoch + 1, 'loss': avg_loss}, best_path)
            print(f"  New best (loss={best_loss:.4f}) -> {best_path}")

    final_path = os.path.join(CONFIG['output_dir'], 'lcam_yolox_v5_finetuned_final.pth')
    torch.save({'model': model.state_dict(), 'epoch': CONFIG['epochs'], 'loss': avg_loss}, final_path)
    print("\n" + "=" * 70)
    print("v5 FINE-TUNING COMPLETE")
    print("=" * 70)
    print(f"Best loss: {best_loss:.4f}")
    print(f"Best  weights: {os.path.join(CONFIG['output_dir'], 'lcam_yolox_v5_finetuned_best.pth')}")
    print(f"Final weights: {final_path}")


if __name__ == '__main__':
    main()
