import matplotlib.pyplot as plt
from config import Config
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping
import os

import segmentation_models_pytorch as smp
import torch
from config import Config


def plot_history(history):

    # Create and save training plot
    plt.figure(figsize=(15, 10))

    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plot IoU score
    plt.subplot(2, 2, 2)
    plt.plot(history.history["iou_score"], label="Training IoU")
    plt.plot(history.history["val_iou_score"], label="Validation IoU")
    plt.title("IoU Score")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.legend()

    # Plot F1 score
    plt.subplot(2, 2, 3)
    plt.plot(history.history["f1_score"], label="Training F1")
    plt.plot(history.history["val_f1_score"], label="Validation F1")
    plt.title("F1 Score")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.legend()

    # Plot Precision and Recall
    plt.subplot(2, 2, 4)
    plt.plot(history.history["precision"], label="Training Precision")
    plt.plot(history.history["val_precision"], label="Validation Precision")
    plt.plot(history.history["recall"], label="Training Recall")
    plt.plot(history.history["val_recall"], label="Validation Recall")
    plt.title("Precision and Recall")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()

    plt.tight_layout()
    plt.savefig(Config.PLOT_PATH)
    print(f"Training metrics plot saved to {Config.PLOT_PATH}")


import glob


def create_callbacks():
    """Create training callbacks to save model only when val_iou_score improves (max 6 checkpoints)"""

    def cleanup_old_checkpoints():
        """Keep only the 6 most recent model checkpoints"""
        checkpoint_files = sorted(
            glob.glob(os.path.join(Config.MODEL_DIR, "model_epoch_*.weights.h5"))
        )
        if len(checkpoint_files) > 6:
            num_to_delete = len(checkpoint_files) - 6
            for file in checkpoint_files[:num_to_delete]:
                print(f"Deleting old checkpoint: {file}")
                os.remove(file)

    class CustomCheckpointCallback(ModelCheckpoint):
        def on_epoch_end(self, epoch, logs=None):
            super().on_epoch_end(epoch, logs)
            cleanup_old_checkpoints()

    # Use the custom callback instead of the regular one

    model_checkpoint_callback = CustomCheckpointCallback(
        filepath=Config.CHECKPOINT_PATH,
        save_weights_only=True,
        monitor="val_iou_score",
        mode="max",
        verbose=1,
        save_best_only=True,
    )

    return [
        model_checkpoint_callback,
        ReduceLROnPlateau(
            monitor="val_loss", patience=3, factor=0.05, verbose=1, min_lr=1e-5
        ),
        CSVLogger(Config.CSV_PATH),
    ]



def defineModel():
    model_name = Config.MODEL.lower()

    if model_name == "unet":
        # Use SMP's pretrained U-Net with ResNet34 encoder
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,              
            activation=None          
        )
    else:
        raise ValueError(
            f"Model '{Config.MODEL}' is not supported. Only 'unet' is implemented."
        )

    return model
