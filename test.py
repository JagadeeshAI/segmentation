#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing script for U-Net segmentation model (weights-only loading + plots)
"""

import glob
import os
import cv2
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import wandb
import pandas as pd

from segmentation_models.metrics import iou_score, f1_score, precision, recall
from segmentation_models.losses import DiceLoss

from config import Config
from model.unet import build_unet
from data_process import load_dataset, get_data_generator
from utils import defineModel

def load_best_model():
    """Rebuild U-Net and load best weights"""
    model = defineModel()
    model = build_unet(Config.INPUT_SHAPE, Config.N_CLASSES)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
        loss=DiceLoss(),
        metrics=[precision, recall, f1_score, iou_score],
    )

    checkpoint_files = glob.glob(
        os.path.join(Config.MODEL_DIR, "model_epoch_*.weights.h5")
    )
    if not checkpoint_files:
        print("No checkpoint weights found. Using final weights.")
        model.load_weights(Config.FINAL_MODEL_PATH)
    else:
        checkpoint_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        model_path = checkpoint_files[-1]
        print(f"Loading model weights from: {model_path}")
        model.load_weights(model_path)

    return model


def visualize_from_generator(model, generator, num_samples=5):
    """Visualize predictions using samples from the test generator"""
    x_batch, y_batch = next(generator)
    for i in range(min(num_samples, len(x_batch))):
        image = x_batch[i]
        true_mask = y_batch[i]

        pred_mask = model.predict(np.expand_dims(image, axis=0), verbose=0)[0]
        pred_mask = (pred_mask > 0.5).astype(np.uint8)

        image_viz = (image * 255).astype(np.uint8)
        true_mask_viz = (true_mask[:, :, 0] * 255).astype(np.uint8)
        pred_mask_viz = (pred_mask[:, :, 0] * 255).astype(np.uint8)

        true_mask_rgb = cv2.cvtColor(true_mask_viz, cv2.COLOR_GRAY2RGB)
        pred_mask_rgb = cv2.cvtColor(pred_mask_viz, cv2.COLOR_GRAY2RGB)
        stacked_image = np.hstack((image_viz, true_mask_rgb, pred_mask_rgb))

        output_path = os.path.join(Config.SAMPLES_DIR, f"sample_{i+1}.jpg")
        cv2.imwrite(output_path, stacked_image)
        wandb.log({f"sample_{i+1}": wandb.Image(stacked_image)})

    print(f"Saved {num_samples} sample visualizations to {Config.SAMPLES_DIR}")


def plot_training_history(csv_path, output_path):
    """Generate training plots from history CSV"""
    if not os.path.exists(csv_path):
        print(f"No training history CSV found at {csv_path}")
        return

    history = pd.read_csv(csv_path)
    plt.figure(figsize=(15, 10))

    # 1. Loss
    if "loss" in history and "val_loss" in history:
        plt.subplot(2, 2, 1)
        plt.plot(history["loss"], label="Train Loss")
        plt.plot(history["val_loss"], label="Val Loss")
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
    else:
        print("Loss columns not found in CSV.")

    # 2. IoU
    if "iou_score" in history and "val_iou_score" in history:
        plt.subplot(2, 2, 2)
        plt.plot(history["iou_score"], label="Train IoU")
        plt.plot(history["val_iou_score"], label="Val IoU")
        plt.title("IoU Score")
        plt.xlabel("Epoch")
        plt.ylabel("IoU")
        plt.legend()
    else:
        print("IoU columns not found in CSV.")

    # 3. F1 Score
    if "f1-score" in history and "val_f1-score" in history:
        plt.subplot(2, 2, 3)
        plt.plot(history["f1-score"], label="Train F1")
        plt.plot(history["val_f1-score"], label="Val F1")
        plt.title("F1 Score")
        plt.xlabel("Epoch")
        plt.ylabel("F1")
        plt.legend()
    else:
        print("F1 Score columns not found in CSV.")

    # 4. Precision & Recall
    if all(
        col in history for col in ["precision", "val_precision", "recall", "val_recall"]
    ):
        plt.subplot(2, 2, 4)
        plt.plot(history["precision"], label="Train Precision")
        plt.plot(history["val_precision"], label="Val Precision")
        plt.plot(history["recall"], label="Train Recall")
        plt.plot(history["val_recall"], label="Val Recall")
        plt.title("Precision & Recall")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.legend()
    else:
        print("Precision/Recall columns not found in CSV.")

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved training plots to {output_path}")
    wandb.log({"training_metrics": wandb.Image(output_path)})


def main():
    print("Starting segmentation model testing...")

    wandb.init(
        project="seg_stats",
        config={
            "input_shape": Config.INPUT_SHAPE,
            "classes": Config.N_CLASSES,
            "architecture": Config.MODEL.upper(),
        },
    )

    aug_images = sorted(glob.glob(Config.DATA_PATH_IMAGES))
    aug_masks = sorted(glob.glob(Config.DATA_PATH_MASKS))
    train_x, val_x, train_y, val_y, test_x, test_y = load_dataset(aug_images, aug_masks)
    print(f"Test set size: {len(test_x)}")

    test_xy_paths = list(zip(test_x, test_y))
    test_generator = get_data_generator(test_xy_paths, batch_size=Config.BATCH_SIZE)

    test_steps = len(test_x) // Config.BATCH_SIZE
    if len(test_x) % Config.BATCH_SIZE != 0:
        test_steps += 1

    model = load_best_model()

    print("Evaluating model on test data...")
    metrics = model.evaluate(test_generator, steps=test_steps, verbose=1)

    print("\nTest Metrics:")
    for name, value in zip(model.metrics_names, metrics):
        print(f"{name}: {value:.4f}")
        wandb.log({f"test_{name}": value})

    print("\nGenerating sample visualizations...")
    visualize_from_generator(model, test_generator, num_samples=5)

    print("\nGenerating training history plots...")
    plot_training_history(Config.CSV_PATH, Config.PLOT_PATH)

    print("Completed model testing.")
    wandb.finish()


if __name__ == "__main__":
    main()
