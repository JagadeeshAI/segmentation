import glob
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import wandb
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping
from segmentation_models.losses import JaccardLoss, DiceLoss, BinaryCELoss
from segmentation_models.metrics import iou_score, f1_score, precision, recall

# Import from local modules
from data_process import load_dataset, get_data_generator
from config import Config
from utils import create_callbacks, defineModel

def load_best_model():
    """Rebuild U-Net and load best weights"""
    model = defineModel()

    if Config.RESUME:
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

def main():
    print(f"Starting pipeline segmentation training using {Config.OUT_DIR}...")

    gpus = tf.config.list_physical_devices("GPU")
    print("*" * 150)
    print("GPUs Available: ", gpus)
    print("*" * 150)

    wandb.init(
        project="segmentation",
        config={
            "learning_rate": Config.LEARNING_RATE,
            "epochs": Config.EPOCHS,
            "batch_size": Config.BATCH_SIZE,
            "input_shape": Config.INPUT_SHAPE,
            "classes": Config.N_CLASSES,
            "architecture": Config.MODEL.upper(),
        },
    )

    # Load and prepare dataset
    aug_images = sorted(glob.glob(Config.DATA_PATH_IMAGES))
    aug_masks = sorted(glob.glob(Config.DATA_PATH_MASKS))
    print(f"Total images: {len(aug_images)}, Total masks: {len(aug_masks)}")

    train_x, val_x, train_y, val_y, test_x, test_y = load_dataset(aug_images, aug_masks)
    print(f"Train: {len(train_x)}, Validation: {len(val_x)}, Test: {len(test_x)}")

    train_X_y_paths = list(zip(train_x, train_y))
    val_X_y_paths = list(zip(val_x, val_y))
    test_X_y_paths = list(zip(test_x, test_y))

    train_generator = get_data_generator(train_X_y_paths, batch_size=Config.BATCH_SIZE)
    val_generator = get_data_generator(val_X_y_paths, batch_size=Config.BATCH_SIZE)
    test_generator = get_data_generator(test_X_y_paths, batch_size=Config.BATCH_SIZE)

    if Config.RESUME:
        print("*" * 50 + " RESUMING " + "*" * 50)
        model = load_best_model()
    else:
        print("*" * 50 + " Starting from scratch " + "*" * 50)
        model = defineModel()

    # Define loss and metrics
    ls = DiceLoss()
    metrics = [precision, recall, f1_score, iou_score]

    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=Config.LEARNING_RATE),
        loss=ls,
        metrics=metrics,
    )

    model.summary()

    train_steps = len(train_x) // Config.BATCH_SIZE + (len(train_x) % Config.BATCH_SIZE != 0)
    val_steps = len(val_x) // Config.BATCH_SIZE + (len(val_x) % Config.BATCH_SIZE != 0)

    callbacks = create_callbacks()
    callbacks.append(
        wandb.keras.WandbCallback(
            monitor="val_iou_score", mode="max", save_model=False, log_weights=True
        )
    )

    print(f"Training model for {Config.EPOCHS} epochs...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=Config.EPOCHS,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        callbacks=callbacks,
        batch_size=Config.BATCH_SIZE,
    )

    model.save(Config.FINAL_MODEL_PATH)
    print(f"Model training completed. Final model saved to {Config.FINAL_MODEL_PATH}")

    print("Evaluating model on test data...")
    test_steps = len(test_x) // Config.BATCH_SIZE + (len(test_x) % Config.BATCH_SIZE != 0)
    print(test_steps)
    model.evaluate(test_generator, batch_size=Config.BATCH_SIZE, steps=test_steps)
    wandb.finish()

if __name__ == "__main__":
    main()
