import numpy as np
import random

import cv2
import librosa
import torch

from .saving import log_path, trainer_paths
from .visualization import TensorboardWriter
from .logger import setup_logger, setup_logging


def hann(iq, window=None):
    """
    Preformes Hann smoothing of 'iq_sweep_burst'.

    Arguments:
      iq {ndarray} -- 'iq_sweep_burst' array
      window -{range} -- range of hann window indices (Default=None)
               if None the whole column is taken

    Returns:
      Regulazied iq in shape - (window[1] - window[0] - 2, iq.shape[1])
    """
    if window is None:
        window = [0, len(iq)]

    N = window[1] - window[0] - 1
    n = np.arange(window[0], window[1])
    n = n.reshape(len(n), 1)
    hannCol = 0.5 * (1 - np.cos(2 * np.pi * (n / N)))
    return (hannCol * iq[window[0] : window[1]])[1:-1]


def fft(iq, axis=0):
    """
    Computes the log of discrete Fourier Transform (DFT).

    Arguments:
    iq_burst -- {ndarray} -- 'iq_sweep_burst' array
    axis -- {int} -- axis to perform fft in (Default = 0)

    Returns:
    log of DFT on iq_burst array
    """
    iq = np.log(np.abs(np.fft.fft(hann(iq), axis=axis)))
    return iq


def center_spectrogram(iq_sweep_burst, doppler_burst):
    spectrogram = fft(iq_sweep_burst, axis=0)
    offset = 63 - int(np.percentile(doppler_burst, 50, interpolation="lower"))
    spectrogram = np.roll(spectrogram, offset, axis=0)
    return spectrogram


def normalize(iq, **kwargs):
    """
    Calculates normalized values for iq_sweep_burst matrix:
    (vlaue-mean)/std.
    """
    m = iq.mean()
    s = iq.std()
    return (iq - m) / s


def minmax_norm(iq, **kwargs):
    return (iq - np.amin(iq)) / (np.amax(iq) - np.amin(iq))


def max_value_on_doppler(iq, doppler_burst):
    """
    Set max value on I/Q matrix using doppler burst vector. 
        
    Arguments:
    iq_burst -- {ndarray} -- 'iq_sweep_burst' array
    doppler_burst -- {ndarray} -- 'doppler_burst' array (center of mass)
                
    Returns:
    I/Q matrix with the max value instead of the original values
    The doppler burst marks the matrix values to change by max value
    """
    iq_max_value = np.max(iq)
    for i in range(iq.shape[1]):
        if doppler_burst[i] >= len(iq):
            continue
        iq[doppler_burst[i], i] = iq_max_value
    return iq


class FreqMask:
    def __init__(self, F, num_masks=1):
        self.F = F
        self.num_masks = num_masks

    def func_images(self, images, random_state, parents, hooks):
        transformed_images = list()
        for img in images:
            img = self.transform(img)
            transformed_images.append(img)
        return transformed_images

    def transform(self, img, **kwargs):
        return freq_mask(img, self.F, self.num_masks)

    def __repr__(self):
        return f"{self.__class__.__name__}(num_masks={self.num_masks}, F={self.F})"


class TimeMask:
    def __init__(self, T, num_masks=1):
        self.T = T
        self.num_masks = num_masks

    def func_images(self, images, random_state, parents, hooks):
        transformed_images = list()
        for img in images:
            img = self.transform(img)
            transformed_images.append(img)
        return transformed_images

    def transform(self, img, **kwargs):
        return time_mask(img, self.T, self.num_masks)

    def __repr__(self):
        return f"{self.__class__.__name__}(num_masks={self.num_masks}, T={self.T})"


class GaussianFilter:
    def __init__(self, kernel_size=(10, 1)):
        self.kernel = np.ones(kernel_size) / np.sum(kernel_size)

    def func_images(self, images, random_state, parents, hooks):
        transformed_images = list()
        for img in images:
            img = self.transform(img)
            transformed_images.append(img)
        return transformed_images

    def transform(self, img, **kwargs):
        return normalize(cv2.filter2D(img, -1, self.kernel))

    def __repr__(self):
        return f"{self.__class__.__name__}(width={self.kernel})"


class Delta:
    def func_images(self, images, random_state, parents, hooks):
        transformed_images = list()
        for img in images:
            img = self.transform(img)
            transformed_images.append(img)
        return transformed_images

    def transform(self, img, **kwargs):
        img[:, :, 1] = normalize(
            librosa.feature.delta(img[:, :, 0], width=9, axis=0, delta=1)
        )
        return img

    def __repr__(self):
        return f"{self.__class__.__name__}(width=9)"


class DeltaDelta:
    def func_images(self, images, random_state, parents, hooks):
        transformed_images = list()
        for img in images:
            img = self.transform(img)
            transformed_images.append(img)
        return transformed_images

    def transform(self, img, **kwargs):
        img[:, :, 1] = normalize(
            librosa.feature.delta(img[:, :, 0], width=9, axis=0, delta=2)
        )
        return img

    def __repr__(self):
        return f"{self.__class__.__name__}(width=9)"


class RollingX:
    def __init__(self, shift=1):
        self.shift = shift

    def func_images(self, images, random_state, parents, hooks):
        transformed_images = list()
        for img in images:
            img = self.transform(img)
            transformed_images.append(img)
        return transformed_images

    def transform(self, img, **kwargs):
        if isinstance(self.shift, tuple):
            shift = np.random.randint(self.shift[0], self.shift[1] + 1)
        else:
            shift = self.shift
        img = np.roll(img, shift, axis=1)
        return img

    def __repr__(self):
        return f"{self.__class__.__name__}(shift={self.shift})"


class RollingY:
    def __init__(self, shift=1):
        self.shift = shift

    def func_images(self, images, random_state, parents, hooks):
        transformed_images = list()
        for img in images:
            img = self.transform(img)
            transformed_images.append(img)
        return transformed_images

    def transform(self, img, **kwargs):
        if isinstance(self.shift, tuple):
            shift = np.random.randint(self.shift[0], self.shift[1] + 1)
        else:
            shift = self.shift
        img = np.roll(img, shift, axis=0)
        return img

    def __repr__(self):
        return f"{self.__class__.__name__}(shift={self.shift})"


class BackgroundBlend:
    def __init__(self, background_directory, alpha=0.8, p=0.5):
        self.background_data = np.float32(np.load(background_directory))
        self.p = p
        self.alpha = alpha

    def func_images(self, images, random_state, parents, hooks):
        transformed_images = list()
        for img in images:
            img = self.transform(img)
            transformed_images.append(img)
        return transformed_images

    def transform(self, img, **kwargs):
        background_idx = np.random.randint(len(self.background_data))
        if np.random.binomial(1, self.p):
            img = (self.alpha * img) + (
                (1 - self.alpha)
                * np.expand_dims(self.background_data[background_idx], axis=-1)
            )
        return img

    def __repr__(self):
        return f"{self.__class__.__name__}(alpha={self.alpha}, p={self.p})"


class BackgroundSuppression:
    def __init__(self):
        pass

    def func_images(self, images, random_state, parents, hooks):
        transformed_images = list()
        for img in images:
            img = self.transform(img)
            transformed_images.append(img)
        return transformed_images

    def transform(self, img, **kwargs):
        img_clone = img.copy()
        img_mean = np.mean(img_clone, axis=(0, 1), keepdims=True)
        img_clone[img_clone < img_mean] = 0
        return img_clone

    def __repr__(self):
        return f"{self.__class__.__name__}"


def freq_mask(spectro, F=3, num_masks=1):
    """Masks frequency values.
    
    Parameters
    ----------
    spectro : numpy.ndarray
        input spectrogram.
    F: number or tuple of number

    
    """
    cloned = spectro.copy()
    num_freq = cloned.shape[0]

    if isinstance(F, tuple):
        F = np.random.randint(F[0], F[1] + 1)

    if isinstance(num_masks, tuple):
        num_masks = np.random.randint(num_masks[0], num_masks[1] + 1)

    for _ in range(0, num_masks):
        f = random.randrange(0, F)
        f_zero = random.randrange(0, num_freq - f)

        # avoids randrange error if values are equal and range is empty
        if f_zero == f_zero + f:
            return cloned

        mask_end = random.randrange(f_zero, f_zero + f)
        cloned[f_zero:mask_end, :, :] = cloned.mean()

    return cloned


def time_mask(spectro, T=3, num_masks=2):
    cloned = spectro.copy()
    num_freq = cloned.shape[1]

    if isinstance(T, tuple):
        T = np.random.randint(T[0], T[1] + 1)

    if isinstance(num_masks, tuple):
        num_masks = np.random.randint(num_masks[0], num_masks[1] + 1)

    for i in range(0, num_masks):
        t = random.randrange(0, T)
        t_zero = random.randrange(0, num_freq - t)

        # avoids randrange error if values are equal and range is empty
        if t_zero == t_zero + t:
            return cloned

        mask_end = random.randrange(t_zero, t_zero + t)
        cloned[:, t_zero:mask_end, :] = cloned.mean()

    return cloned


def moving_average(net1, net2, alpha=1):
    """
    See Also
    --------
    Code taken from https://github.com/timgaripov/swa/blob/master/utils.py
    
    """
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= 1.0 - alpha
        param1.data += param2.data * alpha


def bn_update(loader, model):
    """
    See Also
    --------
    Code taken from https://github.com/timgaripov/swa/blob/master/utils.py
    
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    for input, _ in loader:
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        b = input_var.data.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        model(input_var)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))


def _check_bn(module, flag):
    """
    See Also
    --------
    Code taken from https://github.com/timgaripov/swa/blob/master/utils.py
    
    """
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    """
    See Also
    --------
    Code taken from https://github.com/timgaripov/swa/blob/master/utils.py
    
    """
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    """
    See Also
    --------
    Code taken from https://github.com/timgaripov/swa/blob/master/utils.py
    
    """
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    """
    See Also
    --------
    Code taken from https://github.com/timgaripov/swa/blob/master/utils.py
    
    """
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    """
    See Also
    --------
    Code taken from https://github.com/timgaripov/swa/blob/master/utils.py
    
    """
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]
