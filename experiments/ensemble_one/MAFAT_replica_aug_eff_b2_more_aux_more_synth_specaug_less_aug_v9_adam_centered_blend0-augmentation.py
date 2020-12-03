import abc
from math import sqrt
import torch
import cv2

import imgaug.augmenters as iaa
import torchvision.transforms as T

import sys
from mafat_radar_challenge.utils import (
    FreqMask,
    TimeMask,
    RollingX,
    RollingY,
    Delta,
    DeltaDelta,
    GaussianFilter,
    BackgroundBlend,
    normalize,
    minmax_norm,
    BackgroundSuppression,
)
from albumentations import (
    MultiplicativeNoise,
    MedianBlur,
    GaussianBlur,
    RandomBrightnessContrast,
    CoarseDropout,
    HorizontalFlip,
    OneOf,
    VerticalFlip,
    Lambda,
    Compose,
    Resize,
    Rotate,
    RandomBrightnessContrast,
    ShiftScaleRotate,
    Cutout,
    RandomCrop,
    CenterCrop,
    NoOp,
)

from albumentations.pytorch import ToTensor

sys.path.append("pytorch-auto-augment")


def from_numpy(img):
    return T.ToTensor()(img.copy())


class AugmentationFactoryBase(abc.ABC):
    def build_transforms(self, train):
        return self.build_train() if train else self.build_test()

    @abc.abstractmethod
    def build_train(self):
        pass

    @abc.abstractmethod
    def build_test(self):
        pass


class MNISTTransforms(AugmentationFactoryBase):

    MEANS = [0]
    STDS = [1]

    def build_train(self):
        return T.Compose([T.ToTensor(), T.Normalize(self.MEANS, self.STDS)])

    def build_test(self):
        return T.Compose([T.ToTensor(), T.Normalize(self.MEANS, self.STDS)])


class ImgAugExpTransform(AugmentationFactoryBase):
    def __init__(self, image_size=(126, 32)):
        self.image_size = image_size
        self.background_blend = BackgroundBlend(
            "/home/agarcia/repos/mafat-radar-challenge/mafat_radar_challenge/data/mafat_background_v9_spectrogram.npy",
            alpha=0.8,
            p=0.2,
        )
        self.gaussian_filter = GaussianFilter(kernel_size=(20, 1))
        self.rolling_x = RollingX(shift=(-20, 20))
        self.rolling_y = RollingY(shift=(-20, 20))
        self.delta = Delta()
        self.deltadelta = DeltaDelta()
        self.background_supp = BackgroundSuppression()
        self.freq_mask = FreqMask(F=(5, 25), num_masks=(1, 3))
        self.time_mask = TimeMask(T=(1, 5), num_masks=(1, 10))
        self.aug = Compose(
            [
                Lambda(self.rolling_x.transform),
                Lambda(self.rolling_y.transform),
                # Lambda(self.background_supp.transform),  # Background suppresion
                Lambda(self.background_blend.transform),  # Background blending
                Lambda(minmax_norm),  # This is needed for Noise and Blur addition
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                Rotate(limit=(180, 180), p=0.5),
                # ShiftScaleRotate(
                #     shift_limit=0.1,
                #     scale_limit=0,
                #     rotate_limit=0,
                #     p=0.5,
                #     border_mode=cv2.BORDER_CONSTANT,
                # ),
                # OneOf(
                #     [
                #         MultiplicativeNoise(
                #             multiplier=[0.8, 1.3], elementwise=True, p=0.25
                #         ),
                #         GaussianBlur(p=0.25, blur_limit=(1, 3)),
                #     ]
                # ),
                # RandomBrightnessContrast(
                #     brightness_limit=0.1, contrast_limit=0.1, p=0.1
                # ),
                # Cutout(
                #     num_holes=1,
                #     max_h_size=int(0.2 * self.image_size[0]),
                #     max_w_size=int(0.2 * self.image_size[1]),
                #     p=0.5,
                # ),
                # Lambda(self.delta.transform),
                # Lambda(self.deltadelta.transform),
                # Lambda(self.background_blend.transform),
                # Lambda(self.gaussian_filter.transform),  # Gaussian
                Lambda(self.time_mask.transform),
                Lambda(self.freq_mask.transform),
                # iaa.CenterCropToFixedSize(height=90, width=None),
            ]
        )

    def build_train(self):
        return Compose(
            [
                self.aug,
                # RandomCrop(100, 32),
                Resize(
                    self.image_size[0],
                    self.image_size[1],
                    interpolation=cv2.INTER_CUBIC,
                ),
                Lambda(normalize),
                # T.RandomErasing(),
                ToTensor(),
            ]
        )

    def build_test(self):
        return Compose(
            [
                # iaa.CenterCropToFixedSize(height=90, width=None).augment_image,
                # iaa.Lambda(func_images=self.gaussian_filter.func_images).augment_image,  # Gaussian
                # Lambda(self.background_supp.transform),  # Background suppresion
                Lambda(minmax_norm),
                # CenterCrop(100, 32),
                Resize(
                    self.image_size[0],
                    self.image_size[1],
                    interpolation=cv2.INTER_CUBIC,
                ),
                Lambda(normalize),
                ToTensor(),
            ]
        )


class ImgAugCenteredTransform(AugmentationFactoryBase):
    def __init__(self, image_size=(126, 32)):
        self.image_size = image_size

    def build_train(self):
        return Compose([CenterCrop(64, 32), ToTensor(),])

    def build_test(self):
        return Compose([CenterCrop(64, 32), ToTensor(),])


class ImgAugTTATransform(AugmentationFactoryBase):
    TRANSFORM_LIST = (
        NoOp(p=1),
        HorizontalFlip(p=1),
        VerticalFlip(p=1),
        Rotate(limit=(180, 180), p=1),
        # Lambda(RollingY(shift=(-20, 20)).transform),
        # Lambda(RollingY(shift=(-20, 20)).transform),
    )
    CURR_TRANSFORM = TRANSFORM_LIST[0]

    def __init__(self, image_size=(126, 32)):
        self.image_size = image_size

    def build_train(self):
        pass

    def build_test(self):
        print("Using {}".format(self.CURR_TRANSFORM))
        return Compose(
            [
                Lambda(minmax_norm),
                self.CURR_TRANSFORM,
                Resize(
                    self.image_size[0],
                    self.image_size[1],
                    interpolation=cv2.INTER_CUBIC,
                ),
                Lambda(normalize),
                ToTensor(),
            ]
        )
