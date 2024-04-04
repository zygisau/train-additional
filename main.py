

import os
from lightning import Trainer
from datasets.OSCDLightning import OSCDLightning
from models.SiamLightning import SiamLightning
from transforms.AppendFeatures import AppendFeatures
from transforms.RandomFlip import RandomFlip
from transforms.RandomRot import RandomRot
from utils.dotted import dotted
from utils.parser import get_parser_with_args
from pytorch_lightning.loggers import NeptuneLogger
import torchvision.transforms as tr

neptune_logger = NeptuneLogger(
    project="zygisau/train-additional",
    api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyODJmMDUwMS02NGRmLTRiZGQtYWZlNS0xYmI4ZmFiYjZmMzIifQ==",
    log_model_checkpoints=False,
)


if __name__ == "__main__":
    config_parser, _ = get_parser_with_args('config.json')
    opt = config_parser.parse_args()

    transform = tr.Compose(
        [RandomFlip(), RandomRot()])
    # transform = None
    dataset = OSCDLightning(opt.dataset, opt.batch_size,
                            transform=transform, num_workers=os.cpu_count())

    batch_transform = AppendFeatures(
        opt.feature_model_path, opt.feature_model_checkpoint_path)
    model = SiamLightning(bands='all', lr=opt.lr, transform=batch_transform)

    trainer = Trainer(logger=neptune_logger, max_epochs=opt.max_epochs)
    trainer.fit(model, dataset)
