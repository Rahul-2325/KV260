"""
LCAM-YOLOX v3 Fine-Tuning Script
Fine-tunes from previous checkpoint with new DPU-optimized LCAM

Usage on Kaggle:
  - Upload this folder as a dataset
  - Add D-Fire dataset
  - Run in Kaggle notebook with GPU T4 enabled
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import cv2
import json

# Configuration
sys.path.insert(0, '/kaggle/working/YOLOX')

print("=" * 70)
print("LCAM-YOLOX v3 Fine-Tuning")
print("=" * 70)

# =================================================================
# CONFIG (Edit these as needed)
# =================================================================
CONFIG = {
    # Paths (Kaggle paths)
    'dataset_root': '/kaggle/input/smoke-fire-detection-yolo/data',
    'checkpoint_path': '/kaggle/working/lcam_yolox_wildfire_TRAINED_compat.pth',
    'output_dir': '/kaggle/working/output',
    
    # Model
    'depth': 0.33,
    'width': 0.5,
    'num_classes': 2,  # smoke, fire
    'input_size': 640,
    
    # Fine-tuning specific (lower LR than scratch training)
    'epochs': 20,           # Fine-tune for 20 epochs (was 30 for scratch)
    'batch_size': 16,       # Smaller batch for fine-tuning
    'learning_rate': 1e-4,  # 50x lower than scratch (5e-3)
    'weight_decay': 5e-4,
    'momentum': 0.9,
    
    # Data
    'num_workers': 4,
    
    # Logging
    'print_freq': 50,
    'save_freq': 5,  # Save every N epochs
}

# Print config
print("\nConfiguration:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")


# =================================================================
# DATASET (D-Fire YOLO format)
# =================================================================
class DFireDataset(Dataset):
    """
    D-Fire dataset loader
    
    YOLO format:
      images/IMG.jpg
      labels/IMG.txt   (one detection per line: class x y w h normalized)
    """
    
    def __init__(self, root, split='train', input_size=640, augment=True):
        self.root = Path(root)
        self.split = split
        self.input_size = input_size
        self.augment = augment and (split == 'train')
        
        self.img_dir = self.root / split / 'images'
        self.lbl_dir = self.root / split / 'labels'
        
        # Find all image files
        self.images = sorted(list(self.img_dir.glob('*.jpg')))
        print(f"  {split}: {len(self.images)} images")
    
    def __len__(self):
        return len(self.images)
    
    def load_labels(self, img_path):
        """Load YOLO format labels"""
        lbl_path = self.lbl_dir / (img_path.stem + '.txt')
        
        if not lbl_path.exists():
            return np.zeros((0, 5), dtype=np.float32)  # No detections
        
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
        
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Cannot read: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load labels
        labels = self.load_labels(img_path)
        
        # Resize image to input_size
        h, w = img.shape[:2]
        img = cv2.resize(img, (self.input_size, self.input_size))
        
        # Augmentation
        if self.augment:
            # Random horizontal flip
            if np.random.random() > 0.5:
                img = img[:, ::-1, :].copy()
                if len(labels) > 0:
                    labels[:, 1] = 1 - labels[:, 1]  # Flip x coord
        
        # To tensor: HWC -> CHW, normalize
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        img = torch.from_numpy(img)
        
        return img, labels


# =================================================================
# MAIN TRAINING
# =================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create output dir
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # ====== Build Model ======
    print("\n[1/4] Building LCAM-YOLOX model...")
    from yolox.exp import get_exp
    from yolox.models import YOLOX
    
    # Use yolox_s as base
    exp = get_exp(exp_file=None, exp_name='yolox_s')
    exp.depth = CONFIG['depth']
    exp.width = CONFIG['width']
    exp.num_classes = CONFIG['num_classes']
    exp.input_size = (CONFIG['input_size'], CONFIG['input_size'])
    exp.act = 'lrelu'
    
    model = exp.get_model()
    print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")
    
    # ====== Load Pre-trained Weights ======
    print("\n[2/4] Loading checkpoint for fine-tuning...")
    ckpt = torch.load(CONFIG['checkpoint_path'], map_location='cpu')
    
    state_dict = ckpt['model']
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  Matched: {len(state_dict) - len(unexpected)}")
    print(f"  Missing (will be random init): {len(missing)}")
    print(f"  Unexpected: {len(unexpected)}")
    
    if missing:
        print("\n  New weights to train from scratch:")
        for k in missing[:5]:
            print(f"    - {k}")
    
    model = model.to(device)
    
    # ====== Setup Data ======
    print("\n[3/4] Setting up data...")
    train_ds = DFireDataset(
        CONFIG['dataset_root'],
        split='train',
        input_size=CONFIG['input_size'],
        augment=True
    )
    val_ds = DFireDataset(
        CONFIG['dataset_root'],
        split='valid',
        input_size=CONFIG['input_size'],
        augment=False
    )
    
    def collate_fn(batch):
        imgs, lbls = zip(*batch)
        imgs = torch.stack(imgs)
        # Pad labels to same size
        max_n = max(len(l) for l in lbls)
        max_n = max(max_n, 1)
        padded = np.zeros((len(lbls), max_n, 5), dtype=np.float32)
        for i, l in enumerate(lbls):
            if len(l) > 0:
                padded[i, :len(l)] = l
        return imgs, torch.from_numpy(padded)
    
    train_loader = DataLoader(
        train_ds, batch_size=CONFIG['batch_size'],
        shuffle=True, num_workers=CONFIG['num_workers'],
        collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=CONFIG['batch_size'],
        shuffle=False, num_workers=CONFIG['num_workers'],
        collate_fn=collate_fn, pin_memory=True
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    
    # ====== Optimizer ======
    optimizer = optim.SGD(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        momentum=CONFIG['momentum'],
        weight_decay=CONFIG['weight_decay']
    )
    
    # LR schedule: cosine decay
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG['epochs']
    )
    
    # ====== Training Loop ======
    print("\n[4/4] Starting fine-tuning...")
    best_loss = float('inf')
    
    for epoch in range(CONFIG['epochs']):
        # Train
        model.train()
        epoch_loss = 0
        epoch_start = time.time()
        
        for i, (imgs, targets) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(imgs, targets)
            loss = outputs['total_loss']
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if (i + 1) % CONFIG['print_freq'] == 0:
                avg = epoch_loss / (i + 1)
                lr = optimizer.param_groups[0]['lr']
                elapsed = time.time() - epoch_start
                fps = (i + 1) * CONFIG['batch_size'] / elapsed
                print(f"  Epoch {epoch+1}/{CONFIG['epochs']} "
                      f"[{i+1}/{len(train_loader)}] "
                      f"loss={avg:.4f} lr={lr:.6f} fps={fps:.1f}")
        
        scheduler.step()
        
        avg_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch+1} complete: avg loss={avg_loss:.4f} ({epoch_time:.1f}s)")
        
        # Save checkpoint
        if (epoch + 1) % CONFIG['save_freq'] == 0 or (epoch + 1) == CONFIG['epochs']:
            ckpt_path = os.path.join(CONFIG['output_dir'], f'epoch_{epoch+1}.pth')
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch + 1,
                'loss': avg_loss,
            }, ckpt_path)
            print(f"  Saved: {ckpt_path}")
            
            # Update best
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = os.path.join(CONFIG['output_dir'], 'best.pth')
                torch.save({
                    'model': model.state_dict(),
                    'epoch': epoch + 1,
                    'loss': avg_loss,
                }, best_path)
                print(f"  ✓ New best (loss={best_loss:.4f})")
    
    # Final save
    final_path = os.path.join(CONFIG['output_dir'], 'final.pth')
    torch.save({
        'model': model.state_dict(),
        'epoch': CONFIG['epochs'],
        'loss': avg_loss,
    }, final_path)
    
    print("\n" + "=" * 70)
    print("✓ FINE-TUNING COMPLETE")
    print("=" * 70)
    print(f"Best loss: {best_loss:.4f}")
    print(f"Final saved: {final_path}")


if __name__ == '__main__':
    main()
