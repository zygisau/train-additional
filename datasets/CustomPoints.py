import torch
from torch.utils.data import Dataset
import rasterio
from itertools import combinations


class CustomPoints(Dataset):
    def __init__(self, path, names_file, transform=None):
        self.path = path
        self.transform = transform
        # read folder files
        with open(names_file, 'r') as f:
            self.places = f.readlines()
            # remove whitespace characters like `\n` at the end of each line
            self.places = [x.strip() for x in self.places]
        self.pairs = list(combinations(self.places, 2))

    def read_point(self, point_path):
        src = rasterio.open(point_path)
        return src.read().astype('float32')

    def __len__(self):
        return len(self.pairs) - 1

    def __getitem__(self, idx):
        point_filename = self.pairs[idx]
        points = [self.read_point(point_filename[0]),
                  self.read_point(point_filename[1])]
        # crop to square
        min_size = min(points[0].shape[1], points[1].shape[1],
                       points[0].shape[2], points[1].shape[2])
        points[0] = points[0][:, :224, :224]
        points[1] = points[1][:, :224, :224]
        if self.transform:
            points = self.transform(points)

        return torch.from_numpy(points[0]).float(), torch.from_numpy(points[1]).float()
