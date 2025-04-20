import os
import glob
import matplotlib.pyplot as plt
import torch
import segmentation_models_pytorch as smp
from config import Config


def plot_history(log_path=Config.CSV_PATH, save_path=Config.PLOT_PATH):
    """Plot training history from CSV log file."""
    import pandas as pd

    if not os.path.exists(log_path):
        print(f"Log file {log_path} not found.")
        return

    history = pd.read_csv(log_path)

    plt.figure(figsize=(15, 10))

    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(history["train_loss"], label="Training Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plot IoU score
    if "val_iou" in history.columns:
        plt.subplot(2, 2, 2)
        plt.plot(history["val_iou"], label="Validation IoU")
        plt.title("IoU Score")
        plt.xlabel("Epoch")
        plt.ylabel("IoU")
        plt.legend()

    # Plot Dice score
    if "val_dice" in history.columns:
        plt.subplot(2, 2, 3)
        plt.plot(history["val_dice"], label="Validation Dice")
        plt.title("Dice Score")
        plt.xlabel("Epoch")
        plt.ylabel("Dice")
        plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training metrics plot saved to {save_path}")


def cleanup_old_checkpoints(keep_last=6):
    """Keep only the most recent checkpoints."""
    checkpoint_files = sorted(
        glob.glob(os.path.join(Config.MODEL_DIR, "model_epoch_*.pt"))
    )
    if len(checkpoint_files) > keep_last:
        for file in checkpoint_files[:len(checkpoint_files) - keep_last]:
            print(f"Deleting old checkpoint: {file}")
            os.remove(file)


def get_callbacks():
    """Just a placeholder for PyTorch since we don't use callbacks directly like Keras."""
    # You can integrate your own scheduler or model saving here if needed
    return []


def defineModel():
    """Build and return a segmentation model based on Config.MODEL."""
    model_name = Config.MODEL.lower()

    if model_name == "unet":
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None,  # No sigmoid here; handle in loss or output
        )
    else:
        raise ValueError(
            f"Model '{Config.MODEL}' is not supported. Only 'unet' is implemented."
        )

    return model


def compute_metrics(preds, targets, threshold=0.5):
    preds = (preds > threshold).astype(float)
    targets = targets.astype(float)

    smooth = 1e-8
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()

    dice = (2.0 * intersection + smooth) / (union + smooth)
    iou = (intersection + smooth) / ((union - intersection) + smooth)

    return {
        "dice": dice,
        "iou": iou,
    }



def dice_loss(pred, target, smooth=1e-6):
    """
    Compute Dice Loss for binary segmentation.
    pred: raw logits from the model (before sigmoid)
    target: ground truth mask (same shape)
    """
    pred = torch.sigmoid(pred)
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    return 1 - ((2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth))