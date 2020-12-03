import torch
import random
import numpy as np


def get_random_sample(dataset):
    rnd_idx = random.randint(0, len(dataset) - 1)
    rnd_image = dataset.data[rnd_idx].copy()
    rnd_target = dataset.df.loc[rnd_idx, "target_type"]
    # rnd_image = dataset.transform(rnd_image)
    rnd_image = rnd_image.reshape(rnd_image.shape[0], rnd_image.shape[1])
    rnd_image = np.repeat(rnd_image[:, :, np.newaxis], 3, axis=-1)
    rnd_image = np.float32(rnd_image)
    return rnd_image, rnd_target


class AddMixer:
    def __init__(self, alpha_dist="uniform"):
        assert alpha_dist in ["uniform", "beta"]
        self.alpha_dist = alpha_dist

    def sample_alpha(self):
        if self.alpha_dist == "uniform":
            return random.uniform(0, 0.5)
        elif self.alpha_dist == "beta":
            return np.random.beta(0.4, 0.4)

    def __call__(self, dataset, image, target):
        rnd_image, rnd_target = get_random_sample(dataset)

        alpha = self.sample_alpha()
        image = (1 - alpha) * image + alpha * rnd_image
        target = (1 - alpha) * target + alpha * rnd_target
        return image, target


class SigmoidConcatMixer:
    def __init__(self, sigmoid_range=(3, 12)):
        self.sigmoid_range = sigmoid_range

    def sample_mask(self, size):
        x_radius = random.randint(*self.sigmoid_range)

        step = (x_radius * 2) / size[1]
        x = np.arange(-x_radius, x_radius, step=step)
        y = torch.sigmoid(torch.from_numpy(x)).numpy()
        mix_mask = np.tile(y, (size[0], 1))
        return torch.from_numpy(mix_mask.astype(np.float32))

    def __call__(self, dataset, image, target):
        rnd_image, rnd_target = get_random_sample(dataset)

        mix_mask = self.sample_mask(image.shape[-2:])
        rnd_mix_mask = 1 - mix_mask

        image = mix_mask * image + rnd_mix_mask * rnd_image
        target = target + rnd_target
        target = np.clip(target, 0.0, 1.0)
        return image, target
