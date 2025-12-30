"""
Module to create the dataset for training the U-Net.
"""
import os
import glob
import dataclasses

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from skimage import io

from unet_pytorch.utils import im_to_tensor


class SegmentationDataset(Dataset):
    """Dataset to train the U-Net."""

    def __init__(self, input_files: list, target_files: list) -> None:
        """ Initialize the dataset with input and target files.

        Args:
            input_files: List of input files.
            target_files: List of target files.
        """
        self.x = input_files
        self.y = target_files

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a sample from the dataset.
        Args:
            idx: Index of the sample to return.
        Returns:
            Tuple of input and target tensors.
        """
        input_file = self.x[idx]
        target_file = self.y[idx]

        # Load target mask and normalize from 0-255 to 0-1 for MSE loss
        target_img = io.imread(target_file)
        if target_img.max() > 1.0:
            target_img = target_img.astype(np.float32) / 255.0
        
        x, y = (
            im_to_tensor(input_file).type(torch.float32),
            torch.from_numpy(target_img).type(torch.float32).unsqueeze(0),  # Add channel dimension
        )

        return x, y


@dataclasses.dataclass(init=False)
class UNetDataset():
    """Dataset class for the U-Net model."""

    def __init__(self, in_path: str, out_path: str, val_size: float = 0.1, test_size: float = 0.1) -> None:
        """Initialise the U-Net dataset with a three-way split (train/val/test).

        Args:
            in_path: Path to the input images.
            out_path: Path to the output masks.
            val_size: Fraction of the dataset to use for validation (default 0.1).
            test_size: Fraction of the dataset to use for testing (default 0.1).
        """
        self.in_path = in_path
        self.out_path = out_path

        self.inputs = self._get_filenames(in_path, 'tif')
        self.targets = self._get_filenames(out_path, 'png')

        assert len(self.inputs) == len(self.targets)

        if val_size < 0 or test_size < 0 or (val_size + test_size) >= 1.0:
            raise ValueError("val_size and test_size must be non-negative and sum to less than 1.0")

        # First split off a temporary set that will be split into val and test
        temp_size = val_size + test_size
        if temp_size > 0:
            train_inputs, temp_inputs, train_targets, temp_targets = train_test_split(
                self.inputs, self.targets, test_size=temp_size, random_state=42
            )

            # If both val and test exist, split the temp set accordingly
            if test_size > 0 and val_size > 0:
                relative_test = test_size / temp_size
                val_inputs, test_inputs, val_targets, test_targets = train_test_split(
                    temp_inputs, temp_targets, test_size=relative_test, random_state=42
                )
            elif test_size > 0:
                # all of temp goes to test
                val_inputs, val_targets = [], []
                test_inputs, test_targets = temp_inputs, temp_targets
            else:
                # all of temp goes to val
                val_inputs, val_targets = temp_inputs, temp_targets
                test_inputs, test_targets = [], []
        else:
            # No val/test split; all data is for training
            train_inputs, train_targets = self.inputs, self.targets
            val_inputs, val_targets = [], []
            test_inputs, test_targets = [], []

        self.train_dataset = SegmentationDataset(train_inputs, train_targets)
        self.val_dataset = SegmentationDataset(val_inputs, val_targets)
        self.test_dataset = SegmentationDataset(test_inputs, test_targets)

    def _get_filenames(self, base_path: str, ext: str) -> list:
        """Get a list of files with a specific extension.
        
        Args:
            base_path: Path that contains the files.
            ext: Desired extension for the files.
        Returns:
            List of filenames with the desired extension.
        """
        filenames = glob.glob(os.path.join(base_path, '*.' + ext))
        filenames.sort()
        return filenames

    @property
    def get_train_dataset(self) -> SegmentationDataset:
        """Get the training dataset."""
        return self.train_dataset

    @property
    def get_val_dataset(self) -> SegmentationDataset:
        """Get the validation dataset."""
        return self.val_dataset

    @property
    def get_test_dataset(self) -> SegmentationDataset:
        """Get the test dataset."""
        return self.test_dataset


class UNetDataLoader:
    """Data loader for the U-Net model."""

    def __init__(self, dataset: UNetDataset, batch_size: int, num_workers: int = 0):
        """Initialise the U-Net data loader.

        Args:
            dataset: Dataset object containing the training and validation datasets.
            batch_size: Batch size for the data loader.
            num_workers: Number of workers for the data loader.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_loader = DataLoader(
            dataset.train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True
        )
        self.val_loader = DataLoader(
            dataset.val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True
        )
        self.test_loader = DataLoader(
            dataset.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True
        )

    @property
    def get_train_loader(self) -> DataLoader:
        """Get the training data loader."""
        return self.train_loader

    @property
    def get_val_loader(self) -> DataLoader:
        """Get the validation data loader."""
        return self.val_loader

    @property
    def get_test_loader(self) -> DataLoader:
        """Get the test data loader."""
        return self.test_loader
