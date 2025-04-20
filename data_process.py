import os
import cv2
import numpy as np
import glob
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from config import Config

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def load_dataset(images, masks, random_state=360):
    images = sorted(images)
    masks = sorted(masks)
    # Using fixed random state for dataset splitting
    train_x, test_x, train_y, test_y = train_test_split(
        images, masks, test_size=0.25, random_state=random_state, shuffle=True
    )
    train_x, val_x, train_y, val_y = train_test_split(
        train_x, train_y, test_size=0.15, random_state=random_state, shuffle=True
    )

    # Sort the lists to ensure consistent ordering
    train_x, train_y = zip(*sorted(zip(train_x, train_y)))
    val_x, val_y = zip(*sorted(zip(val_x, val_y)))
    test_x, test_y = zip(*sorted(zip(test_x, test_y)))

    # Convert back to lists
    train_x, train_y = list(train_x), list(train_y)
    val_x, val_y = list(val_x), list(val_y)
    test_x, test_y = list(test_x), list(test_y)

    return train_x, val_x, train_y, val_y, test_x, test_y


def get_data_generator(samples, batch_size):
    while True:
        for offset in range(0, len(samples), batch_size):
            batch_samples = samples[offset : offset + batch_size]
            X_train = []
            Y_train = []

            for batch_sample in batch_samples:
                # Load and normalize image
                X_image = cv2.imread(batch_sample[0])[:, :, ::-1]  # BGR to RGB
                X_image = X_image / 255.0
                X_image = X_image.astype(np.float32)

                # Load and process mask
                Y_image = cv2.imread(batch_sample[1], 0)  # Grayscale
                Y_image = (Y_image < 128).astype(
                    np.float32
                )  # Binary mask: 1 = foreground
                Y_image = np.expand_dims(Y_image, axis=-1)  # (H, W, 1)

                X_train.append(X_image)
                Y_train.append(Y_image)

            yield np.array(X_train), np.array(Y_train)


def main():
    """Main execution function."""
    # Load image paths
    aug_images = sorted(glob.glob(Config.DATA_PATH_IMAGES))
    aug_masks = sorted(glob.glob(Config.DATA_PATH_MASKS))
    print(f"Total images: {len(aug_images)}, Total masks: {len(aug_masks)}")

    # Split dataset
    train_x, val_x, train_y, val_y, test_x, test_y = load_dataset(aug_images, aug_masks)
    print(f"Train: {len(train_x)}, Validation: {len(val_x)}, Test: {len(test_x)}")

    # Create path pairs
    train_X_y_paths = list(zip(train_x, train_y))
    val_X_y_paths = list(zip(test_x, test_y))  # Note: Using test_x for validation
    test_X_y_paths = list(zip(val_x, val_y))  # Note: Using val_x for testing

    # Create data generators
    train_generator = get_data_generator(train_X_y_paths, batch_size=Config.BATCH_SIZE)
    val_generator = get_data_generator(val_X_y_paths, batch_size=Config.BATCH_SIZE)
    test_generator = get_data_generator(test_X_y_paths, batch_size=Config.BATCH_SIZE)

    # Test generators by getting one batch from each
    x_train, y_train = next(train_generator)
    x_val, y_val = next(val_generator)
    x_test, y_test = next(test_generator)

    # Print shapes to verify
    print(f"Training batch shape: {x_train.shape}, {y_train.shape}")
    print(f"Validation batch shape: {x_val.shape}, {y_val.shape}")
    print(f"Testing batch shape: {x_test.shape}, {y_test.shape}")


if __name__ == "__main__":
    main()
