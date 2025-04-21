import os
import glob
import json
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from config import Config
from data_process import SegmentationDataset, load_dataset
from utils import defineModel, compute_metrics, dice_loss, print_model_stats

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for images, masks in tqdm(loader, desc="Training"):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)

        if Config.MODEL.lower() == "fcn":
            outputs = outputs["out"]

        if outputs.shape[-2:] != masks.shape[-2:]:
            outputs = torch.nn.functional.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validating"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            if Config.MODEL.lower() == "fcn":
                outputs = outputs["out"]

            if outputs.shape[-2:] != masks.shape[-2:]:
                outputs = torch.nn.functional.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)

            loss = criterion(outputs, masks)
            val_loss += loss.item()

            preds = torch.sigmoid(outputs).cpu().numpy()
            targets = masks.cpu().numpy()
            all_preds.append(preds)
            all_targets.append(targets)

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    metrics = compute_metrics(all_preds, all_targets)
    return val_loss / len(loader), metrics

def save_checkpoint(model, path, metadata):
    torch.save({
        "model_state_dict": model.state_dict(),
        "metadata": metadata
    }, path)

def load_checkpoint(path, model):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    metadata = checkpoint.get("metadata", {})
    return model, metadata

def save_checkpoint_metadata(json_path, metadata):
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata updated at {json_path}")

def main():
    print(f"Starting segmentation training using {Config.OUT_DIR}...")
    wandb.init(
        project="segmentation-pytorch",
        config={k: getattr(Config, k) for k in dir(Config) if not k.startswith("__")},
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    images = sorted(glob.glob(Config.DATA_PATH_IMAGES))
    masks = sorted(glob.glob(Config.DATA_PATH_MASKS))
    train_x, val_x, train_y, val_y, test_x, test_y = load_dataset(images, masks)

    train_loader = DataLoader(SegmentationDataset(train_x, train_y), batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SegmentationDataset(val_x, val_y), batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(SegmentationDataset(test_x, test_y), batch_size=Config.BATCH_SIZE, shuffle=False)

    best_dice = 0.0
    best_iou = 0.0
    start_epoch = 1
    model = defineModel().to(device)

    sample_image = SegmentationDataset(train_x, train_y)[0][0]
    input_res = tuple(sample_image.shape)
    print_model_stats(model, input_res)

    latest_checkpoint_path = None

    if Config.RESUME and os.path.exists(Config.CHECKPOINT_META_PATH):
        with open(Config.CHECKPOINT_META_PATH, "r") as f:
            resume_info = json.load(f)
        latest_checkpoint_path = resume_info["best_checkpoint"]
        print(f"Resuming from checkpoint: {latest_checkpoint_path}")
        model, metadata = load_checkpoint(latest_checkpoint_path, model)
        best_iou = metadata.get("best_iou", 0.0)
        best_dice = metadata.get("best_dice", 0.0)
        start_epoch = metadata.get("epoch", 1) + 1
        print(f"→ Resumed at epoch {start_epoch}, Best IoU={best_iou:.4f}, Best Dice={best_dice:.4f}")
    else:
        print("Starting from scratch.")

    optimizer = optim.SGD(model.parameters(), lr=Config.LEARNING_RATE, momentum=0.9)
    criterion = dice_loss

    for epoch in range(start_epoch, Config.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{Config.EPOCHS}")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = validate(model, val_loader, criterion, device)

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **val_metrics,
        })

        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            best_dice = val_metrics["dice"]

            epoch_tag = f"{epoch:03d}"
            iou_tag = f"{val_metrics['iou']:.4f}".replace(".", "_")
            model_name = Config.MODEL.lower()
            checkpoint_name = f"{model_name}_ep{epoch_tag}_iou{iou_tag}.pth"
            checkpoint_path = os.path.join(Config.MODEL_DIR, checkpoint_name)

            metadata = {
                "epoch": epoch,
                "best_iou": best_iou,
                "best_dice": best_dice,
                "best_checkpoint": checkpoint_path
            }

            save_checkpoint(model, checkpoint_path, metadata)
            save_checkpoint_metadata(Config.CHECKPOINT_META_PATH, metadata)
            print(f"✔️ Model saved: {checkpoint_path}")

            checkpoint_files = sorted(
                glob.glob(os.path.join(Config.MODEL_DIR, f"{model_name}_ep*.pth")),
                key=os.path.getmtime
            )
            if len(checkpoint_files) > 6:
                to_delete = checkpoint_files[:len(checkpoint_files) - 6]
                for file_path in to_delete:
                    print(f"Deleting old checkpoint: {os.path.basename(file_path)}")
                    os.remove(file_path)

    print("Evaluating on test set...")
    _, test_metrics = validate(model, test_loader, criterion, device)
    print(f"Test metrics: {test_metrics}")
    wandb.log({"test_" + k: v for k, v in test_metrics.items()})
    wandb.finish()

if __name__ == "__main__":
    main()
