import os
import glob
import cv2
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import wandb
from thop import profile
from tqdm import tqdm
import torch.nn.functional as F

from config import Config
from utils import defineModel
from data_process import SegmentationDataset, load_dataset
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score



def load_best_model(model, checkpoint_dir):
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


import time
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
from tqdm import tqdm

def evaluate(model, loader, device):
    model.eval()
    precisions, recalls, f1s, ious = [], [], [], []
    total_images = 0

    print("⚡ Using batch‑wise evaluate() with unconditional interpolation")
    # ensure all pending CUDA work is done before timing
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)
            total_images += images.size(0)

            # forward
            outputs = model(images)

            # unpack model‐specific outputs
            m = Config.MODEL.lower()
            if m == "fcn":
                outputs = outputs["out"]
            elif m == "segformer":
                outputs = outputs.logits

            # resize raw logits to match mask H×W
            outputs = F.interpolate(
                outputs,
                size=masks.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

            # handle binary vs. multi‐class:
            if outputs.shape[1] > 1:
                # multi‐class → collapse to foreground/background
                pred_ids = torch.argmax(outputs, dim=1)          # [B, H, W]
                preds    = (pred_ids != 0).float().unsqueeze(1)  # [B,1,H,W]
            else:
                # single‐channel binary
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()

            # pull to CPU and flatten
            gt_flat = masks.cpu().numpy().ravel()
            pr_flat = preds.cpu().numpy().ravel()

            # batch metrics (zero_division=0 safe)
            precisions.append(precision_score(gt_flat, pr_flat, zero_division=0))
            recalls.append   (recall_score   (gt_flat, pr_flat, zero_division=0))
            f1s.append       (f1_score       (gt_flat, pr_flat, zero_division=0))
            ious.append      (jaccard_score  (gt_flat, pr_flat, zero_division=0))

    # ensure all CUDA work is done before stopping the clock
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - start_time
    fps = total_images / elapsed if elapsed > 0 else float('inf')

    print(f"⚡ Throughput: {fps:.2f} images/sec over {total_images} samples")

    # average across batches
    return {
        "precision": float(np.mean(precisions)),
        "recall":    float(np.mean(recalls)),
        "f1":        float(np.mean(f1s)),
        "iou":       float(np.mean(ious)),
        "fps":       float(fps),
    }



def visualize_predictions(model, loader, device, save_dir, num_samples=5):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    images, masks = next(iter(loader))
    images, masks = images.to(device), masks.to(device)

    with torch.no_grad():
        outputs = model(images)

        if Config.MODEL.lower() == "fcn":
            outputs = outputs["out"]
        elif Config.MODEL.lower() == "segformer":
            outputs = outputs.logits

        # Resize prediction to match mask shape
        if outputs.shape[-2:] != masks.shape[-2:]:
            outputs = torch.nn.functional.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)

        preds = torch.sigmoid(outputs).cpu().numpy()

    for i in range(min(num_samples, len(images))):
        # Convert image tensor to uint8 format
        img = (images[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

        # Get binary ground truth mask and invert it: foreground (pipe) will be white
        true_mask = (masks[i].cpu().numpy()[0] > 0.5).astype(np.uint8)
        true_mask = (1 - true_mask) * 255  # Invert to make foreground white

        # Get binary predicted mask and invert it
        pred_mask = (preds[i][0] > 0.5).astype(np.uint8)
        pred_mask = (1 - pred_mask) * 255  # Invert to make foreground white

        # Convert grayscale masks to 3-channel RGB for visualization
        true_rgb = cv2.cvtColor(true_mask, cv2.COLOR_GRAY2RGB)
        pred_rgb = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2RGB)

        # Stack input image, true mask, and predicted mask side-by-side
        stacked = np.hstack((img, true_rgb, pred_rgb))

        # Save and log the visualization
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

    if "train_loss" in history and "val_loss" in history:
        plt.subplot(2, 2, 1)
        plt.plot(history["train_loss"], label="Train Loss")
        plt.plot(history["val_loss"], label="Val Loss")
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

    if "val_iou" in history:
        plt.subplot(2, 2, 2)
        plt.plot(history["val_iou"], label="Val IoU")
        plt.title("IoU")
        plt.xlabel("Epoch")
        plt.ylabel("IoU")
        plt.legend()

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
    print(device,"used device is")
    # exit()

    images = sorted(glob.glob(Config.DATA_PATH_IMAGES))
    masks = sorted(glob.glob(Config.DATA_PATH_MASKS))
    train_x, val_x, train_y, val_y, test_x, test_y = load_dataset(images, masks)
    print(f"Test set size: {len(test_x)}")

    test_dataset = SegmentationDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    model = defineModel().to(device)
    if Config.FineTuned:
        model = load_best_model(model, Config.MODEL_DIR)

    # metrics = evaluate(model, test_loader, device)
    # print("\nTest Metrics:")
    # for k, v in metrics.items():
    #     print(f"{k}: {v:.4f}")
    #     wandb.log({f"test_{k}": v})

    # dummy_input = torch.randn(1, 3, 256, 320).to(device)
    # flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    # print(f"\nModel Params: {params/1e6:.2f}M | FLOPs: {flops/1e9:.2f} GFLOPs")
    # wandb.log({"Params (M)": params / 1e6, "FLOPs (G)": flops / 1e9})

    # if Config.MODEL.lower() == "deeplabv3+":
    #     print("\nSummary Table:")
    #     print("{:<12} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format("Model", "Precision", "Recall", "F1 Score", "IoU", "Params", "FLOPs"))
    #     print("{:<12} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.2f} {:<10.2f}".format(
    #         Config.MODEL,
    #         metrics.get("precision", 0.0),
    #         metrics.get("recall", 0.0),
    #         metrics.get("f1", 0.0),
    #         metrics.get("iou", 0.0),
    #         params / 1e6,
    #         flops / 1e9
    #     ))

    print("\nGenerating sample visualizations...")
    visualize_predictions(model, test_loader, device, Config.SAMPLES_DIR, num_samples=5)

    print("\nGenerating training history plots...")
    plot_training_history(Config.CSV_PATH, Config.PLOT_PATH)

    print("Testing complete.")
    wandb.finish()


if __name__ == "__main__":
    main()