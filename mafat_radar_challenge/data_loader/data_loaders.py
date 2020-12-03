import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets

from mafat_radar_challenge.base import DataLoaderBase
from mafat_radar_challenge.data_loader import dataset


class MnistDataLoader(DataLoaderBase):
    """
    MNIST data loading demo using DataLoaderBase
    """

    def __init__(
        self,
        transforms,
        data_dir,
        batch_size,
        shuffle,
        validation_split,
        nworkers,
        train=True,
    ):
        self.data_dir = data_dir

        self.train_dataset = datasets.MNIST(
            self.data_dir,
            train=train,
            download=True,
            transform=transforms.build_transforms(train=True),
        )
        self.valid_dataset = (
            datasets.MNIST(
                self.data_dir,
                train=False,
                download=True,
                transform=transforms.build_transforms(train=False),
            )
            if train
            else None
        )

        self.init_kwargs = {"batch_size": batch_size, "num_workers": nworkers}
        super().__init__(self.train_dataset, shuffle=shuffle, **self.init_kwargs)

    def split_validation(self):
        if self.valid_dataset is None:
            return None
        else:
            return DataLoader(self.valid_dataset, **self.init_kwargs)


class MAFATTrainDataLoader(DataLoaderBase):
    """
    Data loader for images and labels of moles, like the ones of SIIM competition.
    """

    def __init__(
        self,
        transforms,
        sampler,
        mixer,
        data_dir,
        csv_dir,
        batch_size,
        shuffle,
        nworkers,
        train=True,
        use_metadata=False,
    ):
        self.data_dir = data_dir
        self.csv_dir = csv_dir

        self.train_dataset = dataset.MAFATDataset(
            self.data_dir,
            self.csv_dir,
            train=train,
            transform=transforms.build_transforms(train=train),
            mixer=mixer,
            use_metadata=use_metadata,
        )

        self.init_kwargs = {
            "batch_size": batch_size,
            "num_workers": nworkers,
            "pin_memory": True,
        }
        sampler = sampler(self.train_dataset) if sampler is not None else None
        super().__init__(
            self.train_dataset, shuffle=shuffle, sampler=sampler, **self.init_kwargs
        )


class MAFATValDataLoader(DataLoaderBase):
    """
    Data loader for images and labels of moles, like the ones of SIIM competition.
    """

    def __init__(
        self,
        transforms,
        sampler,
        data_dir,
        csv_dir,
        batch_size,
        shuffle,
        nworkers,
        train=False,
        use_metadata=False,
    ):
        self.data_dir = data_dir
        self.csv_dir = csv_dir

        self.valid_dataset = dataset.MAFATDataset(
            self.data_dir,
            self.csv_dir,
            train=train,
            transform=transforms.build_transforms(train=train),
            mixer=None,
            use_metadata=use_metadata,
        )

        self.init_kwargs = {
            "batch_size": batch_size,
            "num_workers": nworkers,
            "pin_memory": True,
        }
        super().__init__(self.valid_dataset, **self.init_kwargs)


class MAFATTestDataLoader(DataLoaderBase):
    """
    Data loader for MAFAT.
    """

    def __init__(
        self,
        transforms,
        data_dir,
        batch_size,
        nworkers,
        csv_dir=None,
        train=False,
        use_metadata=False,
    ):
        self.dataset = dataset.MAFATTestDataset(
            data_dir,
            csv_dir,
            transform=transforms.build_transforms(train=False),
            use_metadata=use_metadata,
        )

        self.init_kwargs = {"batch_size": batch_size, "num_workers": nworkers}
        super().__init__(self.dataset, **self.init_kwargs)
