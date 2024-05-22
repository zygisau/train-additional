import os
import lightning as L
import torch

from datasets.MSBC import MSBC
from datasets.OSCD import OSCDDataset
from torch.utils.data import DataLoader

from transforms.Normalise import Normalise
from utils.dotted import dotted
import torchvision.transforms as tr


class MSBCLightning(L.LightningDataModule):
    def __init__(self, root, file_name, img_size, batch_size: int, transform=None, num_workers=None):
        super().__init__()
        self.root = root
        self.file_name = file_name
        self.img_size = img_size
        self.batch_size = batch_size
        self.transform = transform
        self.num_workers = num_workers

    def setup(self, stage):
        self.test_msbc = MSBC(self.root, self.file_name, self.img_size, transform=tr.Compose([Normalise()]))

    def test_dataloader(self):
        return DataLoader(self.test_msbc, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.test_msbc, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def weights(self):
        return self.test_msbc.weights

    def __len__(self) -> int:
        return len(self.test_msbc)
