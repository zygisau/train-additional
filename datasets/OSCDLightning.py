import os
import lightning as L

from datasets.OSCD import OSCDDataset
from torch.utils.data import DataLoader

from utils.dotted import dotted


class OSCDLightning(L.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, transform=None, num_workers=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transform
        self.num_workers = num_workers

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

        self.oscd = OSCDDataset(
            obj.labels, obj.data, fname=obj.file, patch_side=224, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.oscd, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.oscd, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.oscd, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.oscd, batch_size=self.batch_size, num_workers=self.num_workers)

    def __len__(self) -> int:
        return len(self.oscd)
