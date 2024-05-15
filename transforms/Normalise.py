import numpy as np
import torch


class Normalise(object):
    """Normalise the multispectral image.
    """

    def __init__(self):
        pass

    def __call__(self, samples):
        img1, img2, mask = samples

        img1 = (img1 - img1.mean()) / img1.std()
        img2 = (img2 - img2.mean()) / img2.std()

        return img1, img2, mask
