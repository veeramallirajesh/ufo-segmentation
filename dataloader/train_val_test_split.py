# -*- coding: utf-8 -*-
"""
@author: Rajesh

"""

from pathlib import Path
from torch.utils.data import (
    SubsetRandomSampler,
    DataLoader,
    RandomSampler,
    SequentialSampler,
)
from .data_utils import SubsetSequentialSampler
import numpy as np


# If we did split the data before already - no indices needed
class ExistingTrainValTestSplit:
    def __init__(self, dataset_train, dataset_val, dataset_test):
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.dataset_test = dataset_test

    def make_dataloaders(self, batch_size, workers=4):
        train_loader = DataLoader(
            self.dataset_train, batch_size=batch_size, num_workers=workers, shuffle=True
        )
        validation_loader = DataLoader(
            self.dataset_val, batch_size=batch_size, num_workers=workers, shuffle=True
        )
        test_loader = DataLoader(self.dataset_test, batch_size=1)

        return train_loader, validation_loader, test_loader

    def split_exists(self):
        return (
            self.dataset_train is not None
            and self.dataset_val is not None
            and self.dataset_test is not None
        )

    def get_indices(self, test_val_train):
        if test_val_train == "test":
            return np.arange(len(self.dataset_test))
        elif test_val_train == "val":
            return np.arange(len(self.dataset_val))
        elif test_val_train == "train":
            return np.arange(len(self.dataset_train))

        raise ValueError("test_val_train")


class TrainValTestSplit:
    def __init__(self, indices_dir, dataset):
        self.indices_dir = Path(indices_dir)
        self.dataset = dataset

    def make_data_split(self, val_split=0.2, test_split=0.1, shuffle_train_val=False):
        self.indices_dir.mkdir(parents=True, exist_ok=True)

        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        split_train_val = int(np.floor(test_split * dataset_size))
        split_train = int(np.floor(val_split * dataset_size))
        test_indices, train_val_indices = (
            indices[:split_train_val],
            indices[split_train_val:],
        )

        if shuffle_train_val:
            np.random.shuffle(train_val_indices)

        val_indices, train_indices = (
            train_val_indices[:split_train],
            train_val_indices[split_train:],
        )

        print(
            f"{len(test_indices)} test_indices, {len(val_indices)} val_indices, {len(train_indices)} train_indices"
        )

        np.save(self.indices_dir / "test.npy", test_indices)
        np.save(self.indices_dir / "val.npy", val_indices)
        np.save(self.indices_dir / "train.npy", train_indices)

    def make_dataloaders(self, batch_size, workers=4):
        test_indices = np.load(self.indices_dir / "test.npy")
        val_indices = np.load(self.indices_dir / "val.npy")
        train_indices = np.load(self.indices_dir / "train.npy")

        s_train = SubsetRandomSampler(train_indices)
        s_valid = SubsetRandomSampler(val_indices)
        s_test = SubsetSequentialSampler(test_indices)  # dont shuffle the test set

        train_loader = DataLoader(
            self.dataset, batch_size=batch_size, sampler=s_train, num_workers=workers
        )
        validation_loader = DataLoader(
            self.dataset, batch_size=batch_size, sampler=s_valid, num_workers=workers
        )
        test_loader = DataLoader(
            self.dataset, batch_size=1, sampler=s_test
        )  # Batch size for evaluation is 1.

        return train_loader, validation_loader, test_loader

    def make_dataloader(
        self, batch_size, test_val_train, random_sampling=True, workers=4
    ):
        indices = np.load(self.indices_dir / f"{test_val_train}.npy")

        if random_sampling:
            sampler = SubsetRandomSampler(indices)
        else:
            sampler = SubsetSequentialSampler(indices)

        return DataLoader(
            self.dataset, batch_size=batch_size, sampler=sampler, num_workers=workers
        )

    def get_indices(self, test_val_train: str = None):
        indices = np.load(self.indices_dir / f"{test_val_train}.npy")
        return indices

    def split_exists(self):
        return self.indices_dir.exists()
