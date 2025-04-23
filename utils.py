import torch
import torch.nn as nn
import numpy as np
from ptflops import get_model_complexity_info
import segmentation_models_pytorch as smp
from torchvision import models as tv_models
from torchvision.models.segmentation import fcn_resnet50
from transformers import SegformerConfig, SegformerForSemanticSegmentation
from config import Config
from typing import Tuple

class BasicSegNet(nn.Module):
    def __init__(self, num_classes: int = 1):
        super().__init__()
        vgg_features = tv_models.vgg16_bn(pretrained=True).features
        # Encoder: up to pool5
        self.encoder = nn.Sequential(*list(vgg_features)[:33])
        # Symmetric decoder
        channels = [512, 256, 128, 64]
        layers = []
        in_ch = channels[0]
        for out_ch in channels:
            layers += [
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2,
                                   padding=1, output_padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ]
            in_ch = out_ch
        layers.append(nn.ConvTranspose2d(in_ch, num_classes,
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def defineModel() -> nn.Module:
    """Instantiate segmentation model based on Config.MODEL."""
    name = Config.MODEL.lower()
    if name == "unet":
        return smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None
        )
    if name == "fcn":
        m = fcn_resnet50(pretrained=True)
        m.classifier[4] = nn.Conv2d(512, 1, kernel_size=1)
        m.aux_classifier = None
        return m
    if name == "segnet":
        return BasicSegNet(num_classes=1)
    if name == "pspnet":
        return smp.PSPNet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None
        )
    if name == "deeplabv3+":
        return smp.DeepLabV3Plus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None
        )
    if name == "segformer":
        cfg = SegformerConfig.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512",
            num_labels=1
        )
        m = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512",
            config=cfg,
            ignore_mismatched_sizes=True
        )
        # replace classifier head
        m.classifier = nn.Conv2d(cfg.decoder_hidden_size, 1, kernel_size=1)
        return m
    raise ValueError(f"Unsupported model '{Config.MODEL}'. "
                     "Choose from unet, fcn, segnet, pspnet, deeplabv3+, segformer.")


def compute_metrics(preds: np.ndarray,
                    targets: np.ndarray,
                    threshold: float = 0.5,
                    eps: float = 1e-7) -> dict:
    """
    Compute common binary segmentation metrics.
    Returns precision, recall, f1, iou, dice.
    """
    p = (preds > threshold).astype(np.uint8)
    t = (targets > threshold).astype(np.uint8)

    tp = np.sum(p & t)
    fp = np.sum(p & ~t)
    fn = np.sum(~p & t)

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)

    return dict(precision=precision, recall=recall,
                f1=f1, iou=iou, dice=dice)


def dice_loss(pred: torch.Tensor,
              target: torch.Tensor,
              smooth: float = 1e-6) -> torch.Tensor:
    """
    Dice loss for binary masks.
    Expects raw logits in `pred`.
    """
    p = torch.sigmoid(pred).view(pred.size(0), -1)
    t = target.view(target.size(0), -1).float()
    inter = (p * t).sum(dim=1)
    return 1 - ((2 * inter + smooth) /
                (p.sum(dim=1) + t.sum(dim=1) + smooth)).mean()


def print_model_stats(model: nn.Module,
                      input_res: Tuple[int, int]):
    """
    Print trainable parameters (in millions) and FLOPs (in millions)
    for a single input of shape (3, H, W).
    """
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params/1e6:.2f}M")

    try:
        device = 0 if torch.cuda.is_available() else "cpu"
        macs, _ = get_model_complexity_info(
            model.to(device),
            (3, *input_res),
            as_strings=False,
            print_per_layer_stat=False
        )
        print(f"FLOPs: {macs/1e6:.2f}M")
    except Exception as e:
        print("FLOPs calculation failed:", e)
