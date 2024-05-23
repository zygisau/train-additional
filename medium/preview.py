from matplotlib import pyplot
import numpy as np
import rasterio
from rasterio.plot import show
import argparse as ag
from PIL import Image

if __name__ == '__main__':
    # get file from argparse
    parser = ag.ArgumentParser(description='Preview')
    parser.add_argument('--file', type=str, help='Path to file', required=True)

    opt = parser.parse_args()
    dataset: rasterio.DatasetReader = rasterio.open(opt.file)
    data = dataset.read([4, 3, 2])
    norm = (data * (255 / np.max(data))).astype(np.uint8)
    print(norm.shape)
    # norm to pillow
    show(norm)
    pyplot.show()

    # img = Image.fromarray(norm.transpose(1, 2, 0))
    # img.show()
    # get filename from opt.file
    # new_filename = opt.file.split('.')[0]
    # new_filename = f'{new_filename}.png'
    # pyplot.imsave(new_filename, norm.transpose(1, 2, 0))
