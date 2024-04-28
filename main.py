

import os
import signal
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
from lightning.pytorch.plugins.environments import SLURMEnvironment
from lightning.pytorch.callbacks import ModelCheckpoint

neptune_logger = NeptuneLogger(
    project="zygisau/train-additional",
    api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyODJmMDUwMS02NGRmLTRiZGQtYWZlNS0xYmI4ZmFiYjZmMzIifQ==",
    log_model_checkpoints=True,
)


if __name__ == "__main__":
    config_parser, _ = get_parser_with_args('config.json')
    opt = config_parser.parse_args()

    transform = tr.Compose(
        [RandomFlip(), RandomRot()])
    # transform = None
    dataset = OSCDLightning(opt.dataset, opt.batch_size,
                            transform=transform, num_workers=2)
    checkpoints = [
        ModelCheckpoint(auto_insert_metric_name=True, monitor='metrics_train_prec', save_top_k=3, save_last=True, every_n_epochs=1),
        ModelCheckpoint(auto_insert_metric_name=True, monitor='metrics_train_acc', save_top_k=3, save_last=True, every_n_epochs=1),
        ModelCheckpoint(auto_insert_metric_name=True, monitor='metrics_valid_prec', save_top_k=3, save_last=True, every_n_epochs=1),
        ModelCheckpoint(auto_insert_metric_name=True, monitor='metrics_valid_acc', save_top_k=3, save_last=True, every_n_epochs=1),
        ModelCheckpoint(auto_insert_metric_name=True, monitor='metrics_train_prec', save_last=True, every_n_train_steps=1000),
        ModelCheckpoint(auto_insert_metric_name=True, monitor='metrics_train_acc', save_last=True, every_n_train_steps=1000),
        ModelCheckpoint(auto_insert_metric_name=True, monitor='metrics_valid_prec', save_last=True, every_n_train_steps=1000),
        ModelCheckpoint(auto_insert_metric_name=True, monitor='metrics_valid_acc', save_last=True, every_n_train_steps=1000)
    ]
    
    print("=== Best models ===")
    print([c.best_model_path for c in checkpoints])
    print("===================")

    batch_transform = AppendFeatures(
        opt.feature_model_path, opt.feature_model_checkpoint_path)
    model = SiamLightning(bands='all', lr=opt.lr, transform=batch_transform)

    trainer = Trainer(logger=neptune_logger, max_epochs=opt.max_epochs, accelerator='gpu', devices=1, plugins=[SLURMEnvironment(
        requeue_signal=signal.SIGHUP)], callbacks=checkpoints, default_root_dir="/scratch/lustre/home/zyau5516/source/train-additional/.neptune/None/version_None/checkpoints", resume_from_checkpoint="/scratch/lustre/home/zyau5516/source/train-additional/.neptune/None/version_None/checkpoints/epoch=0-step=27000.ckpt")
    trainer.fit(model, dataset)
