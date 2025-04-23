import os
import glob
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from config import Config
from data_process import SegmentationDataset, load_dataset
from utils import defineModel, compute_metrics, dice_loss, print_model_stats


def _extract_logits(output):
    """
    Handle different model heads: FCN returns a dict, SegFormer returns .logits.
    """
    if isinstance(output, dict) and "out" in output:
        return output["out"]
    return getattr(output, "logits", output)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for imgs, masks in tqdm(loader, desc="Train"):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()

        out = _extract_logits(model(imgs))
        if out.shape[-2:] != masks.shape[-2:]:
            out = F.interpolate(out, size=masks.shape[-2:], mode="bilinear", align_corners=False)

        loss = criterion(out, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Validate"):
            imgs, masks = imgs.to(device), masks.to(device)

            out = _extract_logits(model(imgs))
            if out.shape[-2:] != masks.shape[-2:]:
                out = F.interpolate(out, size=masks.shape[-2:], mode="bilinear", align_corners=False)

            val_loss += criterion(out, masks).item()

            all_preds.append(torch.sigmoid(out).cpu().numpy())
            all_targets.append(masks.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    metrics = compute_metrics(preds, targets)

    return val_loss / len(loader), metrics


def save_checkpoint(model, path, metadata):
    torch.save({
        "model_state_dict": model.state_dict(),
        "metadata": metadata
    }, path)


def load_checkpoint(path, model):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt.get("metadata", {})


def main():
    # — init experiment
    wandb.init(
        project="segmentation-pytorch",
        config={k: getattr(Config, k) for k in dir(Config) if not k.startswith("__")}
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # — prepare data
    imgs = sorted(glob.glob(Config.DATA_PATH_IMAGES))
    masks = sorted(glob.glob(Config.DATA_PATH_MASKS))
    assert len(imgs) == len(masks), "Image/mask count mismatch"

    train_imgs, val_imgs, train_masks, val_masks, test_imgs, test_masks = load_dataset(imgs, masks)

    train_loader = DataLoader(SegmentationDataset(train_imgs, train_masks),
                              batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(SegmentationDataset(val_imgs,   val_masks),
                              batch_size=Config.BATCH_SIZE)
    test_loader  = DataLoader(SegmentationDataset(test_imgs,  test_masks),
                              batch_size=Config.BATCH_SIZE)

    # — build model & stats
    model = defineModel().to(device)
    sample, _ = SegmentationDataset(train_imgs, train_masks)[0]
    print_model_stats(model, tuple(sample.shape[1:]))

    # — resume?
    best_iou, best_dice, start_epoch = 0.0, 0.0, 1
    if Config.RESUME and os.path.exists(Config.CHECKPOINT_META_PATH):
        with open(Config.CHECKPOINT_META_PATH) as f:
            info = json.load(f)
        model, meta = load_checkpoint(info["best_checkpoint"], model)
        best_iou   = meta.get("best_iou", 0.0)
        best_dice  = meta.get("best_dice", 0.0)
        start_epoch = meta.get("epoch", 1) + 1
        print(f"Resuming at epoch {start_epoch} (IoU={best_iou:.4f}, Dice={best_dice:.4f})")
    else:
        print("Starting from scratch")

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=Config.LEARNING_RATE,
                                momentum=0.9)
    criterion = dice_loss

    # — train/validate loop
    for epoch in range(start_epoch, Config.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{Config.EPOCHS}")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = validate(model, val_loader, criterion, device)

        wandb.log({"epoch": epoch,
                   "train_loss": train_loss,
                   "val_loss": val_loss,
                   **val_metrics})

        if val_metrics["iou"] > best_iou:
            best_iou, best_dice = val_metrics["iou"], val_metrics.get("dice", best_dice)
            name = f"{Config.MODEL}_ep{epoch:03d}_iou{best_iou:.4f}".replace(".", "_") + ".pth"
            path = os.path.join(Config.MODEL_DIR, name)

            save_checkpoint(model, path, {
                "epoch": epoch,
                "best_iou": best_iou,
                "best_dice": best_dice,
                "best_checkpoint": path
            })
            with open(Config.CHECKPOINT_META_PATH, "w") as f:
                json.dump({
                    "epoch": epoch,
                    "best_iou": best_iou,
                    "best_dice": best_dice,
                    "best_checkpoint": path
                }, f, indent=4)
            print(f"✔️  Saved checkpoint: {path}")

    # — final test evaluation
    print("\nTesting...")
    _, test_metrics = validate(model, test_loader, criterion, device)
    print(f"Test metrics: {test_metrics}")
    wandb.log({f"test_{k}": v for k, v in test_metrics.items()})
    wandb.finish()


if __name__ == "__main__":
    main()
