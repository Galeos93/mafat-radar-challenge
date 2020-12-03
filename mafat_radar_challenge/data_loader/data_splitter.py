import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from sklearn.model_selection import StratifiedKFold, GroupKFold
from mafat_radar_challenge.data_loader import dataset


class KFoldSplitter:
    """Split a dataset into validation and training datasets."""

    def __init__(self, group_idx, n_splits=5):
        self.group_idx = group_idx
        self.n_splits = n_splits
        self.shuffle = True

    def validation_split(self, annotations_df):
        """Takes annotations DataFrame and return indexes that
        corresponds to validation and training data.

        Parameters
        ----------
        annotations_df : pandas.DataFrame
            DataFrame with data annotations.

        Returns
        -------
        train_idx : list of int
            indexes of the dataframe that corresponds to training set.
        val_idx : list of int
            indexes of the dataframe that corresponds to validation set.
        """
        group_kfold = GroupKFold(n_splits=self.n_splits)
        indexes = list(range(len(annotations_df)))
        if self.shuffle:
            np.random.shuffle(indexes)
        kfold_iterable = group_kfold.split(
            indexes, indexes, annotations_df.loc[indexes, "track_id"]
        )
        group_list = list(kfold_iterable)
        train_idx, val_idx = (
            group_list[self.group_idx][0],
            group_list[self.group_idx][1],
        )
        return (
            np.array(indexes)[train_idx].tolist(),
            np.array(indexes)[val_idx].tolist(),
        )


class MetadataSplitter:
    """Split a dataset into validation and training datasets with stratified k fold."""

    def __init__(self, tuple_splitter):
        self.tuple_splitter = tuple_splitter

    def validation_split(self, annotations_df):
        """Takes annotations DataFrame and return indexes that
        corresponds to validation and training data.

        Parameters
        ----------
        annotations_df : pandas.DataFrame
            DataFrame with data annotations.

        Returns
        -------
        train_idx : list of int
            indexes of the dataframe that corresponds to training set.
        val_idx : list of int
            indexes of the dataframe that corresponds to validation set.
        """
        indexes = list(range(len(annotations_df)))
        key, value = self.tuple_splitter
        train_idx = annotations_df[annotations_df[key] != value].index
        val_idx = annotations_df[annotations_df[key] == value].index
        return (
            np.array(indexes)[train_idx].tolist(),
            np.array(indexes)[val_idx].tolist(),
        )


class TrainTestSplitter:
    """Split a dataset into validation and training datasets."""

    def __init__(self, validation_fraction):
        self.validation_fraction = validation_fraction
        self.shuffle = True

    def validation_split(self, annotations_df):
        """Takes annotations DataFrame and return indexes that
        corresponds to validation and training data.

        Parameters
        ----------
        annotations_df : pandas.DataFrame
            DataFrame with data annotations.

        Returns
        -------
        train_idx : list of int
            indexes of the dataframe that corresponds to training set.
        val_idx : list of int
            indexes of the dataframe that corresponds to validation set.
        """
        indexes = list(range(len(annotations_df)))
        train_indexes, val_indexes = sklearn_train_test_split(
            indexes, test_size=self.validation_fraction, random_state=1234
        )
        return list(train_indexes), list(val_indexes)


class MergedDataSplitter:
    """Splits data so validation fold ONLY contains stratified-by-target ISIC2020 data"""

    def __init__(self, validation_fraction):
        self.validation_fraction = validation_fraction

    def validation_split(self, annotations_df):
        """Takes annotations DataFrame and return indexes that
        corresponds to validation and training data.

        Parameters
        ----------
        annotations_df : pandas.DataFrame
            DataFrame with data annotations.

        Returns
        -------
        train_idx : list of int
            indexes of the dataframe that corresponds to training set.
        val_idx : list of int
            indexes of the dataframe that corresponds to validation set.
            This indexes only point to ISIC2020 data.
        """

        isic_2020_df = annotations_df[annotations_df.source == "ISIC20"]
        external_data = annotations_df[annotations_df.source != "ISIC20"]
        indexes = isic_2020_df.index.tolist()
        train_idx, val_idx = sklearn_train_test_split(
            indexes,
            stratify=isic_2020_df.target,
            test_size=self.validation_fraction,
            shuffle=True,
            random_state=1234,
        )
        train_idx += external_data.index.tolist()
        return train_idx, val_idx
