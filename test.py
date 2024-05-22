import matplotlib.pyplot as plt
import torch
import torchvision.transforms as tr
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger

from context import Context
from datasets.MSBCLightning import MSBCLightning
from datasets.OSCDLightning import OSCDLightning
from models.SiamLightning import SiamLightning
# from models.SiamLightningOld import SiamLightning
# from models.SiamLightning_sigmoid import SiamLightningSigmoid
from transforms.AppendFeatures import AppendFeatures
from transforms.ColorJitter import ColorJitter
from transforms.Normalise import Normalise
from transforms.RandomNoise import RandomNoise
from utils.parser import get_parser_with_args

neptune_logger = NeptuneLogger(
    project="zygisau/train-additional",
    api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyODJmMDUwMS02NGRmLTRiZGQtYWZlNS0xYmI4ZmFiYjZmMzIifQ==",
    log_model_checkpoints=True,
)


if __name__ == "__main__":
    config_parser, _ = get_parser_with_args('config.json')
    # config_parser, _ = get_parser_with_args('config_msbc.json')
    opt = config_parser.parse_args()
    neptune_logger.log_hyperparams(opt)

    transform = tr.Compose(
        [Normalise()])
    dataset = OSCDLightning(opt.dataset, opt.batch_size,
                            transform=transform, num_workers=4)
    # dataset = MSBCLightning(opt.root, opt.file_name, 224,
    #                         opt.batch_size, num_workers=4)

    batch_transform = AppendFeatures(
        opt.feature_model_path, opt.feature_model_checkpoint_path)
    context = Context(log_images=False)
    model = SiamLightning(bands='all', lr=opt.lr, transform=batch_transform,
                          model_checkpoint=opt.siam_checkpoint_path, get_weights=dataset.weights, context=context)

    # last_checkpoint = "c:\\Users\\zygimantas\\Downloads\\fc_siam_diff_final.tar"
    last_checkpoint = ".neptune/Untitled/TRAIN-150/checkpoints/last-v7.ckpt"
    checkpoint = torch.load(last_checkpoint)
    if 'state_dict' in checkpoint:
        weights = {k: v for k,
                   v in checkpoint["state_dict"].items() if k.startswith("model.")}
        # rename all keys to remove the "model." prefix
        weights = {k[6:]: v for k, v in weights.items()}
        model.model.load_state_dict(weights)
    else:
        weights = {k: v for k,
                   v in checkpoint.items()}
        weights = {'model.'+k: v for k, v in weights.items()}
        model.load_state_dict(weights)

    trainer = Trainer(logger=neptune_logger, max_epochs=opt.max_epochs, accelerator='gpu', devices=1, fast_dev_run=True,
                      resume_from_checkpoint=last_checkpoint)

    context.log_images = False
    context.log_test_roc = True
    trainer.test(model, dataset)
