import os
import numpy as np
from torch.utils.data import Dataset
import rasterio
from rasterio.enums import Resampling


class MSBC(Dataset):
    def __init__(self, root, file_name, img_size, transform=None):
        self.root = root
        self.file_name = file_name
        self.img_size = img_size
        self.transform = transform

        with open(self.file_name, 'r') as f:
            self.images = f.readlines()
            self.images = [x.strip() for x in self.images]

    def get_rgb1_path(self, image_name):
        return os.path.join(self.root, 'rgb1', image_name + '.tif')

    def get_rgb2_path(self, image_name):
        return os.path.join(self.root, 'rgb2', image_name + '.tif')

    def get_opt_path(self, image_name):
        return os.path.join(self.root, 'opt', image_name + '.tif')

    def get_label_path(self, image_name):
        return os.path.join(self.root, 'label', image_name + '.tif')

    def load_data(self, image_name):
        with rasterio.open(self.get_rgb1_path(image_name)) as src:
            rgb1 = src.read(
                out_shape=(
                    src.count,
                    self.img_size,
                    self.img_size
                ),
                resampling=Resampling.bilinear
            )

        with rasterio.open(self.get_rgb2_path(image_name)) as src:
            rgb2 = src.read(
                out_shape=(
                    src.count,
                    self.img_size,
                    self.img_size
                ),
                resampling=Resampling.bilinear
            )

        with rasterio.open(self.get_opt_path(image_name)) as src:
            opt = src.read(
                out_shape=(
                    src.count,
                    self.img_size,
                    self.img_size
                ),
                resampling=Resampling.bilinear
            )
            # B2,B3,B4,B8,B8A,B11,B12
            opt1 = opt[:7]
            opt2 = opt[7:]

        image1 = np.stack([rgb1[0, :, :], rgb1[1, :, :], rgb1[2, :, :], opt1[3, :, :], np.zeros_like(rgb1[0, :, :]), np.zeros_like(rgb1[0, :, :]), np.zeros_like(
            rgb1[0, :, :]), opt1[4, :, :], opt1[5, :, :], opt1[6, :, :], np.zeros_like(rgb1[0, :, :]), np.zeros_like(rgb1[0, :, :]), np.zeros_like(rgb1[0, :, :])], axis=0)
        image2 = np.stack([rgb2[0, :, :], rgb2[1, :, :], rgb2[2, :, :], opt2[3, :, :], np.zeros_like(rgb2[0, :, :]), np.zeros_like(rgb2[0, :, :]), np.zeros_like(
            rgb2[0, :, :]), opt2[4, :, :], opt2[5, :, :], opt2[6, :, :], np.zeros_like(rgb1[0, :, :]), np.zeros_like(rgb1[0, :, :]), np.zeros_like(rgb1[0, :, :])], axis=0)

        with rasterio.open(self.get_label_path(image_name)) as src:
            label = src.read(out_shape=(
                src.count,
                self.img_size,
                self.img_size
            ),
                resampling=Resampling.bilinear
            )
            label = label.squeeze().astype(np.float32)

        return image1, image2, label

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_name = self.images[index]
        load_data = self.load_data(image_name)
        if self.transform:
            load_data = self.transform(load_data)
        return load_data
