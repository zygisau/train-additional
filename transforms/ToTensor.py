import numpy as np
import torch


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, samples):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img1, img2, mask = samples
        img1 = np.array(img1).astype(np.float32).transpose((2, 0, 1))
        img2 = np.array(img2).astype(np.float32).transpose((2, 0, 1))
        # mask = np.expand_dims(np.array(mask).astype(np.float32), axis=0)
        mask = np.array(mask).astype(np.float32)

        img1 = torch.from_numpy(img1).float()
        img2 = torch.from_numpy(img2).float()
        mask = torch.from_numpy(mask).float()

        return img1, img2, mask
