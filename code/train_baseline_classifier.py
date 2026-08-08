"""
Baseline Dry Waste Classifier
==============================
Quick transfer-learning baseline to validate a waste-sorting image dataset.

Expected folder structure:

    data/
        train/
            glass/   *.jpg
            paper/   *.jpg
            metal/   *.jpg
            plastic/ *.jpg
        val/
            glass/ ...
            paper/ ...
            metal/ ...
            plastic/ ...

If you only have one big folder per class (no train/val split yet), run
`split_dataset()` at the bottom first to create the split automatically.

Requirements:
    pip install torch torchvision scikit-learn matplotlib pillow --break-system-packages
"""

import os
import shutil
import random
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
DATA_DIR = "data"          # expects data/train and data/val
RAW_DIR = "raw_data"       # optional: unsplit data (class folders only)
IMG_SIZE = 224
BATCH_SIZE = 128
EPOCHS = 30
LR = 1e-4
VAL_SPLIT = 0.2
SEED = 42
OUTPUT_DIR = "output"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------
# OPTIONAL: split raw class folders into train/val
# ----------------------------------------------------------------------
def split_dataset(raw_dir=RAW_DIR, out_dir=DATA_DIR, val_split=VAL_SPLIT, seed=SEED):
    """
    If your images are currently just:
        raw_data/glass/*.jpg
        raw_data/paper/*.jpg
        raw_data/metal/*.jpg
        raw_data/plastic/*.jpg
    this creates data/train/<class> and data/val/<class> with a random split.
    """
    random.seed(seed)
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)

    if not raw_dir.exists():
        print(f"[split_dataset] {raw_dir} not found, skipping split.")
        return

    EXCLUDE_CLASSES = {"glass"}

    classes = [d.name for d in raw_dir.iterdir() if d.is_dir() and d.name not in EXCLUDE_CLASSES]
    print(f"[split_dataset] Found classes: {classes}")
    if EXCLUDE_CLASSES:
        print(f"[split_dataset] Excluding: {sorted(EXCLUDE_CLASSES)}")

    for cls in classes:
        images = list((raw_dir / cls).glob("*"))
        images = [p for p in images if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        random.shuffle(images)

        n_val = int(len(images) * val_split)
        val_imgs = images[:n_val]
        train_imgs = images[n_val:]

        for split_name, split_imgs in [("train", train_imgs), ("val", val_imgs)]:
            dest = out_dir / split_name / cls
            dest.mkdir(parents=True, exist_ok=True)
            for img_path in split_imgs:
                shutil.copy(img_path, dest / img_path.name)

        print(f"[split_dataset] {cls}: {len(train_imgs)} train / {len(val_imgs)} val")


# ----------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------
def get_dataloaders(data_dir=DATA_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tfms)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_tfms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=6, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True, persistent_workers=True)

    return train_loader, val_loader, train_ds.classes


# ----------------------------------------------------------------------
# MODEL
# ----------------------------------------------------------------------
def build_model(num_classes, unfreeze_last_n_blocks=2):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # Freeze everything first
    for param in model.features.parameters():
        param.requires_grad = False

    # Unfreeze the last N feature blocks so the model can adapt
    # mid/low-level features to this dataset, not just the classifier head.
    if unfreeze_last_n_blocks > 0:
        total_blocks = len(model.features)
        for block in model.features[total_blocks - unfreeze_last_n_blocks:]:
            for param in block.parameters():
                param.requires_grad = True

    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model.to(DEVICE)


# ----------------------------------------------------------------------
# TRAIN / EVAL LOOPS
# ----------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return running_loss / total, correct / total, all_preds, all_labels


# ----------------------------------------------------------------------
# PLOTTING / REPORTING
# ----------------------------------------------------------------------
def plot_history(history, out_dir):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train Acc")
    axes[1].plot(epochs, history["val_acc"], label="Val Acc")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, out_dir):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    plt.close()


# ----------------------------------------------------------------------
# DATASET DOCUMENTATION EXPORTS
# ----------------------------------------------------------------------
def export_class_distribution(data_dir, out_dir):
    """Bar chart + JSON of image counts per class, per split. Useful for a dataset card."""
    counts = {}
    for split in ("train", "val"):
        split_dir = Path(data_dir) / split
        if not split_dir.exists():
            continue
        for cls_dir in sorted(split_dir.iterdir()):
            if cls_dir.is_dir():
                n = len(list(cls_dir.glob("*")))
                counts.setdefault(cls_dir.name, {})[split] = n

    with open(os.path.join(out_dir, "class_distribution.json"), "w") as f:
        json.dump(counts, f, indent=2)

    classes = list(counts.keys())
    train_counts = [counts[c].get("train", 0) for c in classes]
    val_counts = [counts[c].get("val", 0) for c in classes]

    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, train_counts, width, label="Train")
    ax.bar(x + width / 2, val_counts, width, label="Val")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("Image count")
    ax.set_title("Class distribution")
    ax.legend()

    for i, (t, v) in enumerate(zip(train_counts, val_counts)):
        ax.text(i - width / 2, t + max(train_counts) * 0.01, str(t), ha="center", fontsize=9)
        ax.text(i + width / 2, v + max(train_counts) * 0.01, str(v), ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "class_distribution.png"), dpi=150)
    plt.close()

    print(f"[export_class_distribution] {counts}")
    return counts


def export_sample_grid(data_dir, out_dir, samples_per_class=4, img_size=IMG_SIZE):
    """Grid image showing example samples from each class — good for a dataset card 'preview'."""
    split_dir = Path(data_dir) / "train"
    classes = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])

    fig, axes = plt.subplots(len(classes), samples_per_class,
                              figsize=(samples_per_class * 2.2, len(classes) * 2.2))

    if len(classes) == 1:
        axes = np.expand_dims(axes, axis=0)

    from PIL import Image
    for row, cls in enumerate(classes):
        img_paths = list((split_dir / cls).glob("*"))[:samples_per_class]
        for col in range(samples_per_class):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(img_paths):
                img = Image.open(img_paths[col]).convert("RGB").resize((img_size, img_size))
                ax.imshow(img)
            if col == 0:
                ax.set_ylabel(cls, fontsize=11)
                ax.axis("on")
                ax.set_xticks([])
                ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "sample_grid.png"), dpi=150)
    plt.close()
    print("[export_sample_grid] saved sample_grid.png")


def export_dataset_summary(class_names, counts, best_val_acc, report, out_dir):
    """Single markdown file with the key stats/figures, ready to paste into a dataset card."""
    total_train = sum(v.get("train", 0) for v in counts.values())
    total_val = sum(v.get("val", 0) for v in counts.values())

    lines = [
        "# Dataset Summary (auto-generated)",
        "",
        f"- Classes: {', '.join(class_names)}",
        f"- Total images: {total_train + total_val} ({total_train} train / {total_val} val)",
        "",
        "## Class distribution",
        "",
        "| Class | Train | Val | Total |",
        "|---|---|---|---|",
    ]
    for c in class_names:
        t = counts.get(c, {}).get("train", 0)
        v = counts.get(c, {}).get("val", 0)
        lines.append(f"| {c} | {t} | {v} | {t + v} |")

    lines += [
        "",
        "## Baseline model",
        "",
        "- Architecture: MobileNetV2 (ImageNet-pretrained, frozen backbone, fine-tuned classifier head)",
        f"- Best validation accuracy: {best_val_acc:.4f}",
        "",
        "```",
        report,
        "```",
        "",
        "## Figures",
        "",
        "- `class_distribution.png` — per-class image counts (train/val)",
        "- `sample_grid.png` — example images per class",
        "- `training_curves.png` — loss/accuracy over epochs",
        "- `confusion_matrix.png` — validation confusion matrix",
    ]

    with open(os.path.join(out_dir, "dataset_summary.md"), "w") as f:
        f.write("\n".join(lines))

    print("[export_dataset_summary] saved dataset_summary.md")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # If you have an unsplit raw_data/ folder, uncomment to auto-split:
    # split_dataset()

    print(f"Using device: {DEVICE}")

    train_loader, val_loader, class_names = get_dataloaders()
    print(f"Classes: {class_names}")
    print(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")

    model = build_model(num_classes=len(class_names), unfreeze_last_n_blocks=2)
    criterion = nn.CrossEntropyLoss()

    # Lower LR for the unfrozen backbone layers, higher LR for the new classifier head
    backbone_params = [p for p in model.features.parameters() if p.requires_grad]
    head_params = model.classifier.parameters()
    optimizer = torch.optim.Adam([
        {"params": backbone_params, "lr": LR * 0.1},
        {"params": head_params, "lr": LR},
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    epochs_without_improvement = 0
    EARLY_STOP_PATIENCE = 5

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
              f"{elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pt"))
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(f"Early stopping: no val improvement for {EARLY_STOP_PATIENCE} epochs.")
                break

    # Final report using best model
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pt")))
    _, _, final_preds, final_labels = evaluate(model, val_loader, criterion)

    report = classification_report(final_labels, final_preds, target_names=class_names, digits=3)
    print("\n" + report)

    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
        f.write(report)

    with open(os.path.join(OUTPUT_DIR, "class_names.json"), "w") as f:
        json.dump(class_names, f)

    plot_history(history, OUTPUT_DIR)
    plot_confusion_matrix(final_labels, final_preds, class_names, OUTPUT_DIR)

    # --- Dataset documentation exports (for the dataset card / README) ---
    counts = export_class_distribution(DATA_DIR, OUTPUT_DIR)
    export_sample_grid(DATA_DIR, OUTPUT_DIR)
    export_dataset_summary(class_names, counts, best_val_acc, report, OUTPUT_DIR)

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print(f"Artifacts saved to: {OUTPUT_DIR}/")
    print("  - best_model.pt")
    print("  - classification_report.txt")
    print("  - training_curves.png")
    print("  - confusion_matrix.png")
    print("  - class_names.json")
    print("  - class_distribution.png / .json")
    print("  - sample_grid.png")
    print("  - dataset_summary.md  <- paste this into your dataset card")


if __name__ == "__main__":
    main()
