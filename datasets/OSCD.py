# Functions

import os
from math import ceil, floor

import numpy as np
import torch
from pandas import read_csv
from PIL import Image
from scipy.ndimage import zoom
from skimage import io
from torch.utils.data import Dataset
from tqdm import tqdm

NORMALISE_IMGS = False
FP_MODIFIER = 1  # Tuning parameter, use 1 if unsure
TYPE = 3  # 0: RGB, 1: RGB+NIR, 2: RGB+NIR+SWIR1, 3: All bands


def adjust_shape(I, s):
    """Adjust shape of grayscale image I to s."""

    # crop if necesary
    I = I[:s[0], :s[1]]
    si = I.shape

    # pad if necessary
    p0 = max(0, s[0] - si[0])
    p1 = max(0, s[1] - si[1])

    return np.pad(I, ((0, p0), (0, p1)), 'edge')


def read_sentinel_img(path):
    """Read cropped Sentinel-2 image: RGB bands."""
    im_name = os.listdir(path)[0][:-7]
    r = io.imread(path + im_name + "B04.tif")
    g = io.imread(path + im_name + "B03.tif")
    b = io.imread(path + im_name + "B02.tif")

    I = np.stack((r, g, b), axis=2).astype('float')

    if NORMALISE_IMGS:
        I = (I - I.mean()) / I.std()

    return I


def read_sentinel_img_4(path):
    """Read cropped Sentinel-2 image: RGB and NIR bands."""
    im_name = os.listdir(path)[0][:-7]
    r = io.imread(path + im_name + "B04.tif")
    g = io.imread(path + im_name + "B03.tif")
    b = io.imread(path + im_name + "B02.tif")
    nir = io.imread(path + im_name + "B08.tif")

    I = np.stack((r, g, b, nir), axis=2).astype('float')

    if NORMALISE_IMGS:
        I = (I - I.mean()) / I.std()

    return I


def read_sentinel_img_leq20(path):
    """Read cropped Sentinel-2 image: bands with resolution less than or equals to 20m."""
    im_name = os.listdir(path)[0][:-7]

    r = io.imread(path + im_name + "B04.tif")
    s = r.shape
    g = io.imread(path + im_name + "B03.tif")
    b = io.imread(path + im_name + "B02.tif")
    nir = io.imread(path + im_name + "B08.tif")

    ir1 = adjust_shape(zoom(io.imread(path + im_name + "B05.tif"), 2), s)
    ir2 = adjust_shape(zoom(io.imread(path + im_name + "B06.tif"), 2), s)
    ir3 = adjust_shape(zoom(io.imread(path + im_name + "B07.tif"), 2), s)
    nir2 = adjust_shape(zoom(io.imread(path + im_name + "B8A.tif"), 2), s)
    swir2 = adjust_shape(zoom(io.imread(path + im_name + "B11.tif"), 2), s)
    swir3 = adjust_shape(zoom(io.imread(path + im_name + "B12.tif"), 2), s)

    I = np.stack((r, g, b, nir, ir1, ir2, ir3, nir2,
                 swir2, swir3), axis=2).astype('float')

    if NORMALISE_IMGS:
        I = (I - I.mean()) / I.std()

    return I


def read_sentinel_img_leq60(path):
    """Read cropped Sentinel-2 image: all bands."""
    names = os.listdir(path)
    im_name = ''
    for name in names:
        if name.endswith('.tif'):
            im_name = name[:-7]

    r = io.imread(path + im_name + "B04.tif")
    s = r.shape
    g = io.imread(path + im_name + "B03.tif")
    b = io.imread(path + im_name + "B02.tif")
    nir = io.imread(path + im_name + "B08.tif")

    ir1 = adjust_shape(zoom(io.imread(path + im_name + "B05.tif"), 2), s)
    ir2 = adjust_shape(zoom(io.imread(path + im_name + "B06.tif"), 2), s)
    ir3 = adjust_shape(zoom(io.imread(path + im_name + "B07.tif"), 2), s)
    nir2 = adjust_shape(zoom(io.imread(path + im_name + "B8A.tif"), 2), s)
    swir2 = adjust_shape(zoom(io.imread(path + im_name + "B11.tif"), 2), s)
    swir3 = adjust_shape(zoom(io.imread(path + im_name + "B12.tif"), 2), s)

    uv = adjust_shape(zoom(io.imread(path + im_name + "B01.tif"), 6), s)
    wv = adjust_shape(zoom(io.imread(path + im_name + "B09.tif"), 6), s)
    swirc = adjust_shape(zoom(io.imread(path + im_name + "B10.tif"), 6), s)

    I = np.stack((r, g, b, nir, ir1, ir2, ir3, nir2, swir2,
                 swir3, uv, wv, swirc), axis=2).astype('float')

    if NORMALISE_IMGS:
        I = (I - I.mean()) / I.std()

    return I


def read_sentinel_img_trio(img_path, labels_path, band_mode):
    """Read cropped Sentinel-2 image pair and change map."""
#     read images
    if band_mode == 'rgb':
        I1 = read_sentinel_img(img_path + '/imgs_1/')
        I2 = read_sentinel_img(img_path + '/imgs_2/')
    # elif TYPE == 1:
    #     I1 = read_sentinel_img_4(img_path + '/imgs_1/')
    #     I2 = read_sentinel_img_4(img_path + '/imgs_2/')
    # elif TYPE == 2:
    #     I1 = read_sentinel_img_leq20(img_path + '/imgs_1/')
    #     I2 = read_sentinel_img_leq20(img_path + '/imgs_2/')
    elif band_mode == 'all':
        I1 = read_sentinel_img_leq60(img_path + '/imgs_1/')
        I2 = read_sentinel_img_leq60(img_path + '/imgs_2/')

    cm = io.imread(labels_path + '/cm/cm.png', as_gray=True) != 0

    # crop if necessary
    s1 = I1.shape
    s2 = I2.shape
    s_max = (max(s1[0], s2[0]), max(s1[1], s2[1]), max(s1[2], s2[2]))
    new_I2 = np.pad(
        I2, ((0, 0), (0, s_max[1] - s2[1]), (0, s_max[2] - s2[2])), 'edge')
    new_I1 = np.pad(
        I1, ((0, 0), (0, s_max[1] - s1[1]), (0, s_max[2] - s1[2])), 'edge')

    # skip if shapes are not equal

    return new_I1, new_I2, cm


def reshape_for_torch(I):
    """Transpose image for PyTorch coordinates."""
#     out = np.swapaxes(I,1,2)
#     out = np.swapaxes(out,0,1)
#     out = out[np.newaxis,:]
    return I.transpose((2, 0, 1))


class OSCDDataset(Dataset):
    """Change Detection dataset class, used for both training and test data."""

    def __init__(self, labels_path, path, patch_side=96, stride=None, transform=None, fname=None, band_mode='all'):
        """
        Args:
            root_dir (string): Directory with all the images.
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        # basics
        self.path = path
        self.patch_side = patch_side
        self.transform = transform
        if not stride:
            self.stride = 1
        else:
            self.stride = stride

#         print(path + fname)
        self.names = read_csv(fname).columns
        self.n_imgs = self.names.shape[0]

        n_pix = 0
        true_pix = 0

        # load images
        self.imgs_1 = {}
        self.imgs_2 = {}
        self.change_maps = {}
        self.n_patches_per_image = {}
        self.n_patches = 0
        self.patch_coords = []
        for im_name in tqdm(self.names):
            # load and store each image
            I1, I2, cm = read_sentinel_img_trio(
                path + im_name, labels_path + im_name, band_mode=band_mode)
            self.imgs_1[im_name] = reshape_for_torch(I1)
            self.imgs_2[im_name] = reshape_for_torch(I2)
            self.change_maps[im_name] = cm

            s = cm.shape
            n_pix += np.prod(s)
            true_pix += cm.sum()

            # calculate the number of patches
            s = self.imgs_1[im_name].shape
            n1 = floor((s[1] - self.patch_side + 1) / self.stride)
            n2 = floor((s[2] - self.patch_side + 1) / self.stride)
            n_patches_i = n1 * n2
            self.n_patches_per_image[im_name] = n_patches_i
            self.n_patches += n_patches_i

            # generate path coordinates
            for i in range(n1):
                for j in range(n2):
                    # coordinates in (x1, x2, y1, y2)
                    current_patch_coords = (im_name,
                                            [self.stride*i, self.stride*i + self.patch_side,
                                                self.stride*j, self.stride*j + self.patch_side],
                                            [self.stride*(i + 1), self.stride*(j + 1)])
                    self.patch_coords.append(current_patch_coords)

        self.weights = [FP_MODIFIER * 2 * true_pix /
                        n_pix, 2 * (n_pix - true_pix) / n_pix]

    def get_img(self, im_name):
        return self.imgs_1[im_name], self.imgs_2[im_name], self.change_maps[im_name]

    def __len__(self):
        return self.n_patches

    def __getitem__(self, idx):
        current_patch_coords = self.patch_coords[idx]
        im_name = current_patch_coords[0]
        limits = current_patch_coords[1]
        centre = current_patch_coords[2]

        I1 = self.imgs_1[im_name][:, limits[0]:limits[1],
                                  limits[2]:limits[3]]
        I2 = self.imgs_2[im_name][:, limits[0]:limits[1],
                                  limits[2]:limits[3]]

        if I1.shape[1] < self.patch_side or I1.shape[2] < self.patch_side:
            print(
                f'Problem with image size, image: {im_name}, I1 shape: {I1.shape}, I2 shape {I2.shape}, limits: {limits}')
            print('EXPECT ERROR')
        I1 = np.pad(I1, ((0, 0), (0, self.patch_side -
                    I1.shape[1]), (0, self.patch_side - I1.shape[2])), 'edge')
        I2 = np.pad(I2, ((0, 0), (0, self.patch_side -
                    I2.shape[1]), (0, self.patch_side - I2.shape[2])), 'edge')

        I1 = torch.from_numpy(I1).float()
        I2 = torch.from_numpy(I2).float()

        label = self.change_maps[im_name][limits[0]:limits[1], limits[2]:limits[3]]
        label = np.pad(label, ((0, self.patch_side -
                       label.shape[0]), (0, self.patch_side - label.shape[1])), 'edge')
        label = torch.from_numpy(1*np.array(label)).float()

        sample = [I1, I2, label]
        if self.transform:
            sample = self.transform(sample)

        return sample
