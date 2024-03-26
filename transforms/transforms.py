from torchvision.transforms import transforms

from transforms.FixedResize import FixedResize
from transforms.Normalize import Normalize
from transforms.ToTensor import ToTensor


train_transforms = transforms.Compose([
    # RandomHorizontalFlip(),
    # RandomVerticalFlip(),
    # RandomFixRotate(),
    # RandomRotate(30),
    FixedResize(256),
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensor()])

test_transforms = transforms.Compose([
    FixedResize(256),
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensor()])
