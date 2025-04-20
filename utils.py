import os
import glob
import matplotlib.pyplot as plt
import torch
import segmentation_models_pytorch as smp
from config import Config
import torchvision.models.segmentation as models
import torch.nn as nn
from ptflops import get_model_complexity_info

def defineModel():
    """Build and return a segmentation model based on Config.MODEL."""
    model_name = Config.MODEL.lower()

    if model_name == "unet":
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None,  # We'll apply sigmoid manually
        )

    elif model_name == "fcn":
        model = models.fcn_resnet50(pretrained=True)
        # Replace classifier to output 1 class (for binary segmentation)
        model.classifier[4] = nn.Conv2d(512, 1, kernel_size=1)
        model.aux_classifier[4] = nn.Conv2d(256, 1, kernel_size=1)  # Optional aux head
        model.aux_classifier = None  # Disable aux branch if not needed

    else:
        raise ValueError(
            f"Model '{Config.MODEL}' is not supported. Use 'unet' or 'fcn'."
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

def print_model_stats(model, input_res):
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable Parameters: {num_params / 1e6:.2f}M")

        try:
            with torch.cuda.device(0 if torch.cuda.is_available() else "cpu"):
                macs, _ = get_model_complexity_info(model, input_res, as_strings=False, print_per_layer_stat=False)
                print(f"FLOPs: {macs / 1e6:.2f}M")
        except Exception as e:
            print(f"FLOPs calculation failed: {e}")