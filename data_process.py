import os
import cv2
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from config import Config

class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, normalize=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.normalize = normalize
        self.target_height = 256
        self.target_width = 320

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and convert BGR to RGB
        image = cv2.imread(self.image_paths[idx])[:, :, ::-1].astype(np.float32)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        # Normalize image (0–1 range)
        if self.normalize:
            image /= 255.0

        # Convert image to CHW format
        image = np.transpose(image, (2, 0, 1))  # (C, H, W)
        mask = np.expand_dims((mask < 128).astype(np.float32), axis=0)  # (1, H, W)

        # Padding if needed
        _, h, w = image.shape
        pad_h = self.target_height - h if h < self.target_height else 0
        pad_w = self.target_width - w if w < self.target_width else 0

        if pad_h > 0 or pad_w > 0:
            image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant')
            mask = np.pad(mask, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant')

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)


def load_dataset(image_paths, mask_paths, random_state=360):
    image_paths = sorted(image_paths)
    mask_paths = sorted(mask_paths)

    train_x, test_x, train_y, test_y = train_test_split(
        image_paths, mask_paths, test_size=0.25, random_state=random_state, shuffle=True
    )

    train_x, val_x, train_y, val_y = train_test_split(
        train_x, train_y, test_size=0.15, random_state=random_state, shuffle=True
    )

    return train_x, val_x, train_y, val_y, test_x, test_y

def main():
    # Load paths
    images = sorted(glob.glob(Config.DATA_PATH_IMAGES))
    masks = sorted(glob.glob(Config.DATA_PATH_MASKS))
    print(f"Total images: {len(images)}, Total masks: {len(masks)}")

    # Split data
    train_x, val_x, train_y, val_y, test_x, test_y = load_dataset(images, masks)
    print(f"Train: {len(train_x)}, Val: {len(val_x)}, Test: {len(test_x)}")

    # Create datasets
    train_dataset = SegmentationDataset(train_x, train_y)
    val_dataset = SegmentationDataset(val_x, val_y)
    test_dataset = SegmentationDataset(test_x, test_y)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Get one batch to test
    x_train, y_train = next(iter(train_loader))
    x_val, y_val = next(iter(val_loader))
    x_test, y_test = next(iter(test_loader))

    print(f"Training batch shape: {x_train.shape}, {y_train.shape}")
    print(f"Validation batch shape: {x_val.shape}, {y_val.shape}")
    print(f"Testing batch shape: {x_test.shape}, {y_test.shape}")

if __name__ == "__main__":
    main()
