"""
Training Script — DeepFake Detection
Paper: AI-Driven Detection of Deepfake Content for Enhancing Digital Trust

Training hyperparameters (Section III-C):
  - Optimizer: Adam, lr=1e-4, cosine annealing to 1e-6
  - Batch size: 32
  - Epochs: 50 (custom CNN), 30 (EfficientNet fine-tune)
  - Loss: Binary Cross-Entropy
  - Early stopping patience: 7
  - Hardware: NVIDIA RTX 3080 (10 GB VRAM)
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import cv2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from app import DeepfakeCNN, EfficientNetDetector

# ─── Dataset ───────────────────────────────────────────────────────────
class FaceDataset(Dataset):
    """
    Expects directory structure:
      root/real/  ← real face crops
      root/fake/  ← manipulated face crops
    Augmentation as per paper Section III-B:
      - Horizontal flip
      - ±15° rotation
      - Color jitter
      - Gaussian blur 3×3
      - Random erasing p=0.2
    """
    def __init__(self, root, train=True):
        self.samples = []
        for label, folder in [(0, 'real'), (1, 'fake')]:
            path = os.path.join(root, folder)
            if os.path.exists(path):
                for f in os.listdir(path):
                    if f.lower().endswith(('.jpg','.jpeg','.png')):
                        self.samples.append((os.path.join(path, f), label))

        # ImageNet normalization (paper Section III-B)
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        if train:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),                         # ±15° paper
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),          # paper
                transforms.GaussianBlur(kernel_size=3),               # paper
                transforms.ToTensor(),
                normalize,
                transforms.RandomErasing(p=0.2),                      # paper
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize,
            ])

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img), torch.tensor(label, dtype=torch.float32)


# ─── Training ──────────────────────────────────────────────────────────
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Training on: {device}")

    # Datasets
    train_ds = FaceDataset(args.data_dir, train=True)
    val_ds   = FaceDataset(args.val_dir,  train=False)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=4)  # batch=32 (paper)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=4)
    print(f"[INFO] Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Model
    if args.model == 'cnn':
        model = DeepfakeCNN().to(device)
        n_epochs = 50      # paper: 50 epochs for custom CNN
        print(f"[INFO] Custom CNN — {sum(p.numel() for p in model.parameters()):,} params")
    else:
        model = EfficientNetDetector().to(device)
        n_epochs = 30      # paper: 30 epochs for EfficientNet fine-tuning
        print(f"[INFO] EfficientNet-B4 — {sum(p.numel() for p in model.parameters()):,} params")

    # Optimizer: Adam lr=1e-4 (paper)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Cosine annealing lr → 1e-6 (paper)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

    # Binary cross-entropy loss (paper)
    criterion = nn.BCELoss()

    best_val_auc = 0
    patience_counter = 0
    patience = 7  # early stopping patience (paper)

    for epoch in range(1, n_epochs + 1):
        # ── Train ──
        model.train()
        train_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # ── Validate ──
        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                probs = model(imgs).cpu().squeeze(1).numpy()
                preds = (probs > 0.5).astype(int)
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(labels.numpy().astype(int))

        acc  = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, zero_division=0)
        rec  = recall_score(all_labels, all_preds, zero_division=0)
        f1   = f1_score(all_labels, all_preds, zero_division=0)
        auc  = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0

        avg_loss = train_loss / len(train_loader)
        lr_now = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch:03d}/{n_epochs} | Loss: {avg_loss:.4f} | "
              f"Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | "
              f"F1: {f1:.4f} | AUC: {auc:.4f} | LR: {lr_now:.2e}")

        # Save best model
        if auc > best_val_auc:
            best_val_auc = auc
            save_path = f"../models/{args.model}_best.pth"
            os.makedirs('../models', exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"  → Saved best model (AUC={auc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[INFO] Early stopping at epoch {epoch} (patience={patience})")
                break

    print(f"\n[DONE] Best AUC-ROC: {best_val_auc:.4f}")
    print(f"[INFO] Paper reported: Accuracy=94.2%, AUC-ROC=0.971")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='efficientnet', choices=['cnn','efficientnet'])
    parser.add_argument('--data_dir', default='../data/train')
    parser.add_argument('--val_dir',  default='../data/val')
    args = parser.parse_args()
    train(args)
