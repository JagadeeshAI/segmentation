import os
import glob
import cv2
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import wandb

from config import Config
from utils import defineModel, compute_metrics
from data_process import SegmentationDataset, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm


def load_best_model(model, checkpoint_dir):
    """Load best model from checkpoint directory"""
    checkpoints = sorted(
        glob.glob(os.path.join(checkpoint_dir, "*.pth")),
        key=os.path.getmtime
    )
    if not checkpoints:
        raise FileNotFoundError("No checkpoints found.")
    best_path = checkpoints[-1]
    checkpoint = torch.load(best_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded best model from: {best_path}")
    return model


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)
            outputs = torch.sigmoid(model(images))
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(masks.cpu().numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    return compute_metrics(preds, targets)


def visualize_predictions(model, loader, device, save_dir, num_samples=5):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    images, masks = next(iter(loader))
    images, masks = images.to(device), masks.to(device)

    with torch.no_grad():
        preds = torch.sigmoid(model(images)).cpu().numpy()

    for i in range(min(num_samples, len(images))):
        img = (images[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        true_mask = (masks[i].cpu().numpy()[0] * 255).astype(np.uint8)
        pred_mask = (preds[i][0] > 0.5).astype(np.uint8) * 255

        true_rgb = cv2.cvtColor(true_mask, cv2.COLOR_GRAY2RGB)
        pred_rgb = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2RGB)
        stacked = np.hstack((img, true_rgb, pred_rgb))

        save_path = os.path.join(save_dir, f"sample_{i+1}.jpg")
        cv2.imwrite(save_path, stacked)
        wandb.log({f"sample_{i+1}": wandb.Image(stacked)})

    print(f"Saved {num_samples} visualizations to {save_dir}")


def plot_training_history(csv_path, output_path):
    if not os.path.exists(csv_path):
        print(f"No CSV found at {csv_path}")
        return

    history = pd.read_csv(csv_path)
    plt.figure(figsize=(15, 10))

    # 1. Loss
    if "train_loss" in history and "val_loss" in history:
        plt.subplot(2, 2, 1)
        plt.plot(history["train_loss"], label="Train Loss")
        plt.plot(history["val_loss"], label="Val Loss")
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

    # 2. IoU
    if "val_iou" in history:
        plt.subplot(2, 2, 2)
        plt.plot(history["val_iou"], label="Val IoU")
        plt.title("IoU")
        plt.xlabel("Epoch")
        plt.ylabel("IoU")
        plt.legend()

    # 3. Dice
    if "val_dice" in history:
        plt.subplot(2, 2, 3)
        plt.plot(history["val_dice"], label="Val Dice")
        plt.title("Dice Score")
        plt.xlabel("Epoch")
        plt.ylabel("Dice")
        plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved training plot to {output_path}")
    wandb.log({"training_plot": wandb.Image(output_path)})


def main():
    print("Starting PyTorch segmentation testing...")

    wandb.init(
        project="seg_stats",
        config={
            "input_shape": Config.INPUT_SHAPE,
            "classes": Config.N_CLASSES,
            "architecture": Config.MODEL.upper(),
        },
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    images = sorted(glob.glob(Config.DATA_PATH_IMAGES))
    masks = sorted(glob.glob(Config.DATA_PATH_MASKS))
    train_x, val_x, train_y, val_y, test_x, test_y = load_dataset(images, masks)
    print(f"Test set size: {len(test_x)}")

    test_dataset = SegmentationDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Load model
    model = defineModel().to(device)
    model = load_best_model(model, Config.MODEL_DIR)

    # Evaluate
    metrics = evaluate(model, test_loader, device)
    print("\nTest Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        wandb.log({f"test_{k}": v})

    # Visualize
    print("\nGenerating sample visualizations...")
    visualize_predictions(model, test_loader, device, Config.SAMPLES_DIR, num_samples=5)

    # Plot training history
    print("\nGenerating training history plots...")
    plot_training_history(Config.CSV_PATH, Config.PLOT_PATH)

    print("Testing complete.")
    wandb.finish()


if __name__ == "__main__":
    main()
