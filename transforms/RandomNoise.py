import numpy as np
import torch


class RandomNoise(object):
    """Add random noise to the multispectral image.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, mean=0.5, std=1):
        self.mean = mean
        self.std = std

    def __call__(self, samples):
        img1, img2, mask = samples

        should_apply = np.random.rand() < 0.5
        if should_apply:
            img1 += torch.normal(mean=self.mean,
                                 std=self.std, size=img1.size())
        should_apply = np.random.rand() < 0.5
        if should_apply:
            img2 += torch.normal(mean=self.mean,
                                 std=self.std, size=img2.size())
        return img1, img2, mask
