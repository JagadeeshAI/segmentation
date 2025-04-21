import os
import glob
import matplotlib.pyplot as plt
import torch
import segmentation_models_pytorch as smp
from config import Config
import torchvision.models.segmentation as models
import torch.nn as nn
from ptflops import get_model_complexity_info

import torch.nn as nn
import segmentation_models_pytorch as smp
import torchvision.models.segmentation as models
import torchvision.models as tv_models
from config import Config


class BasicSegNet(nn.Module):
    def __init__(self, num_classes=1):
        super(BasicSegNet, self).__init__()
        vgg = tv_models.vgg16_bn(pretrained=True)
        features = list(vgg.features.children())

        self.encoder = nn.Sequential(*features[:33])  # Up to maxpool5
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, num_classes, 3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def defineModel():
    """Build and return a segmentation model based on Config.MODEL."""
    model_name = Config.MODEL.lower()

    if model_name == "unet":
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None,
        )

    elif model_name == "fcn":
        model = models.fcn_resnet50(pretrained=True)
        model.classifier[4] = nn.Conv2d(512, 1, kernel_size=1)
        model.aux_classifier = None  # Disable aux branch if not needed

    elif model_name == "segnet":
        model = BasicSegNet(num_classes=1)

    elif model_name == "pspnet":
        model = smp.PSPNet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None,
        )

    elif model_name == "deeplabv3+":
        model = smp.DeepLabV3Plus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None,
        )

    else:
        raise ValueError(
            f"Model '{Config.MODEL}' is not supported. Use 'unet', 'fcn', 'segnet', 'pspnet', or 'deeplabv3+'."
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