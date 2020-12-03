import numpy as np
import pandas as pd
from torch.utils.data import Sampler
import logging
import random

logger = logging.getLogger(__file__)
logger.setLevel(logging.WARNING)


class WeightedSourceSampler(Sampler):
    """Sampler that fetches batch of samples with certain amount of classes."""

    def __init__(self, dataset, p=0.2):
        """
        Parameters
        ----------
        dataset : siim.data_loader.siim_dataset.SIIMDataset
            Input dataset to be sampled from.
        """
        self.df = dataset.df
        self.p = p
        pos_segments = self.df[self.df.source == "train"].segment_id
        neg_segments = self.df[self.df.source == "aux"].segment_id
        self.pos_indexes = list(self.df[self.df.segment_id.isin(pos_segments)].index)
        self.neg_indexes = list(self.df[self.df.segment_id.isin(neg_segments)].index)
        super(WeightedSourceSampler, self).__init__(dataset)

    def __iter__(self):
        for _ in range(len(self.df)):
            class_index = np.random.choice(2, p=[1 - self.p, self.p])
            if class_index == 1:
                yield random.choice(self.pos_indexes)
            else:
                yield random.choice(self.neg_indexes)

    def __len__(self):
        return self.df.shape[0]


class WeightedSynthSampler(Sampler):
    """Sampler that fetches batch of samples with certain amount of classes."""

    def __init__(self, dataset, p=0.8):
        """
        Parameters
        ----------
        dataset : siim.data_loader.siim_dataset.SIIMDataset
            Input dataset to be sampled from.
        """
        self.df = dataset.df
        self.p = p
        self.pos_indexes = list(self.df[self.df.source != "synth"].index)
        self.neg_indexes = list(self.df[self.df.source == "synth"].index)
        super(WeightedSynthSampler, self).__init__(dataset)

    def __iter__(self):
        for _ in range(len(self.df)):
            class_index = np.random.choice(2, p=[1 - self.p, self.p])
            if class_index == 1:
                yield random.choice(self.pos_indexes)
            else:
                yield random.choice(self.neg_indexes)

    def __len__(self):
        return self.df.shape[0]


class BalancedTrackSampler(Sampler):
    """Sampler that fetches batch of samples with certain amount of classes."""

    def __init__(self, dataset):
        """
        Parameters
        ----------
        dataset : siim.data_loader.siim_dataset.SIIMDataset
            Input dataset to be sampled from.
        """
        self.df = dataset.df.loc[dataset.data_indexes]
        self.data_indexes_map = {
            data: idx for idx, data in enumerate(dataset.data_indexes)
        }
        self.track_idx_map = dict(self.df.groupby("track_id")["index"].apply(list))
        self.track_ids = list(self.track_idx_map.keys())
        super(BalancedTrackSampler, self).__init__(dataset)

    def __iter__(self):
        for _ in range(len(self.df)):
            track_id = random.choice(self.track_ids)
            data_index = random.choice(self.track_idx_map[track_id])
            yield self.data_indexes_map[data_index]

    def __len__(self):
        return self.df.shape[0]
