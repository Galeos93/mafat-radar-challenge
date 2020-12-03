import os
import numpy as np
import cv2
import pandas as pd
import pickle
from torch.utils.data import Dataset
import torch
from mafat_radar_challenge.data_loader import data_splitter
from mafat_radar_challenge.utils import fft, normalize, center_spectrogram
import zarr
import functools
import librosa


def data_loader_dispatcher(ext):
    def load_pickle(data_path):
        with open(data_path, "rb") as data:
            return pickle.load(data)

    if ext == ".npy":
        return np.load
    elif ext == ".pkl":
        return load_pickle
    elif ext == ".zarr":
        return functools.partial(zarr.open, mode="r")
    else:
        raise ValueError("Extension is not accepted")


class MAFATAdvDataset(Dataset):
    """MAFAT dataset."""

    def __init__(
        self, data_dir, csv_dir, train=True, transform=None, splitter=None,
    ):
        """Init method.

        Parameters
        ----------
        data_dir : str
            Path where data lives.
        csv_dir : str
            Path where data annotations lives.
        transform : callable
            Callable that tranforms an image.
        use_metadata: bool
            Boolean that triggers the use of metadata features.

        """
        self.data_dir = data_dir
        data_loader = data_loader_dispatcher(os.path.splitext(data_dir)[1])
        self.data = data_loader(data_dir)
        self.df = pd.read_csv(csv_dir).reset_index()
        train_idx, val_idx = splitter.validation_split(self.df)
        if train:
            indexes = train_idx
        else:
            indexes = val_idx
        self.df = self.df.loc[indexes]
        self.df = self.df.reset_index()
        self.data = self.data[indexes]

        self.train = train
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = self.df.loc[idx, "target_type"]
        features = self.data[idx]
        features = features.reshape(features.shape[0], features.shape[1])
        features = np.repeat(features[:, :, np.newaxis], 3, axis=-1)
        features = np.float32(features)
        if self.transform:
            features = self.transform(features)
        else:
            features = features
        return features, torch.FloatTensor([label])


class MAFATDataset(Dataset):
    """MAFAT dataset."""

    def __init__(
        self,
        data_dir,
        csv_dir,
        train=True,
        transform=None,
        mixer=None,
        use_metadata=False,
    ):
        """Init method.

        Parameters
        ----------
        data_dir : str
            Path where data lives.
        csv_dir : str
            Path where data annotations lives.
        transform : callable
            Callable that tranforms an image.
        use_metadata: bool
            Boolean that triggers the use of metadata features.

        """
        self.data_dir = data_dir
        data_loader = data_loader_dispatcher(os.path.splitext(data_dir)[1])
        self.data = data_loader(data_dir)
        self.df = pd.read_csv(csv_dir).reset_index()
        self.meta_features = [x for x in self.df.columns if "burst_" in x]
        self.use_metadata = use_metadata
        self.train = train
        self.transform = transform
        self.mixer = mixer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = self.df.loc[idx, "target_type"]
        features = self.data[idx]
        features = features.reshape(features.shape[0], features.shape[1])
        features = np.repeat(features[:, :, np.newaxis], 3, axis=-1)
        features = np.float32(features)
        metadata = np.array(
            self.df.loc[idx, self.meta_features].values, dtype=np.float32
        ).tolist()
        if self.mixer:
            features, label = self.mixer(self, features, label)
        if self.transform:
            features = self.transform(**{"image": features})
            features = features["image"]
        else:
            features = features
        if self.use_metadata:
            return (
                (features, torch.FloatTensor(metadata)),
                torch.FloatTensor([label]),
            )
        return features, torch.FloatTensor([label])


class MAFATTestDataset(Dataset):
    """MAFAT test dataset."""

    def __init__(
        self, data_dir, csv_dir, transform=None, use_metadata=False,
    ):
        """Init method.

        Parameters
        ----------
        data_dir : str
            Path where data lives.
        csv_dir : str
            Path where data annotations lives.

        """
        self.data_dir = data_dir
        data_loader = data_loader_dispatcher(os.path.splitext(data_dir)[1])
        self.data = data_loader(data_dir)
        self.transform = transform
        self.df = pd.read_csv(csv_dir).reset_index()
        self.meta_features = [x for x in self.df.columns if "burst_" in x]
        self.use_metadata = use_metadata

    def __len__(self):
        if os.path.splitext(self.data_dir)[1] == ".pkl":
            return len(self.data["iq_sweep_burst"])
        else:
            return len(self.data)

    def __getitem__(self, idx):
        if os.path.splitext(self.data_dir)[1] == ".pkl":
            features = center_spectrogram(
                self.data["iq_sweep_burst"][idx], self.data["doppler_burst"][idx]
            )
        else:
            features = self.data[idx]
        features = features.reshape(features.shape[0], features.shape[1])
        features = np.repeat(features[:, :, np.newaxis], 3, axis=-1)
        features = np.float32(features)
        metadata = np.array(
            self.df.loc[idx, self.meta_features].values, dtype=np.float32
        ).tolist()
        if self.transform:
            features = self.transform(**{"image": features})
            features = features["image"]
        else:
            features = features
        if self.use_metadata:
            return (features, torch.FloatTensor(metadata))

        return features
