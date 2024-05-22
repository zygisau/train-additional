import os
import lightning as L
import torch

from datasets.OSCD import OSCDDataset
from torch.utils.data import DataLoader

from transforms.Normalise import Normalise
from utils.dotted import dotted
import torchvision.transforms as tr


class OSCDLightning(L.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, transform=None, num_workers=None, band_mode='all'):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transform
        self.num_workers = num_workers
        self.band_mode = band_mode

        self.train = dotted({
            "file": os.path.join(data_dir, 'train.txt'),
            "data": os.path.join(data_dir, 'images') + os.path.sep,
            "labels": os.path.join(data_dir, 'train_labels') + os.path.sep
        })
        self.valid = dotted({
            "file": os.path.join(data_dir, 'valid.txt'),
            "data": os.path.join(data_dir, 'images') + os.path.sep,
            "labels": os.path.join(data_dir, 'test_labels') + os.path.sep
        })
        self.test = dotted({
            "file": os.path.join(data_dir, 'test.txt'),
            "data": os.path.join(data_dir, 'images') + os.path.sep,
            "labels": os.path.join(data_dir, 'test_labels') + os.path.sep
        })

    def setup(self, stage):
        # 'fit', 'validate', 'test', or 'predict'
        if stage == 'fit':
            obj = self.train
        elif stage == 'validate':
            obj = self.valid
        elif stage == 'test' or stage == 'predict':
            obj = self.test
        else:
            raise ValueError(f"Stage {stage} not recognized")

        PATCH_SIDE = 224
        self.train_oscd = OSCDDataset(
            obj.labels, obj.data, fname=obj.file, patch_side=PATCH_SIDE, stride=16, transform=self.transform, band_mode=self.band_mode)
        self.valid_oscd = OSCDDataset(
            obj.labels, obj.data, fname=obj.file, patch_side=PATCH_SIDE, stride=16, transform=tr.Compose([Normalise()]), band_mode=self.band_mode)
        self.test_oscd = OSCDDataset(
            obj.labels, obj.data, fname=obj.file, patch_side=PATCH_SIDE, stride=16, transform=tr.Compose([Normalise()]), band_mode=self.band_mode)

    def train_dataloader(self):
        return DataLoader(self.train_oscd, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.valid_oscd, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_oscd, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.test_oscd, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def weights(self):
        if torch.cuda.is_available():
            return torch.cuda.FloatTensor(self.train_oscd.weights)
        return torch.FloatTensor(self.train_oscd.weights)

    def __len__(self) -> int:
        return len(self.train_oscd)
