import matplotlib.pyplot as plt
import torch
import torchvision.transforms as tr
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger

from context import Context
from datasets.OSCDLightning import OSCDLightning
from models.SiamLightning import SiamLightning
from models.SiamLightning_sigmoid import SiamLightningSigmoid
from transforms.AppendFeatures import AppendFeatures
from transforms.ColorJitter import ColorJitter
from transforms.Normalise import Normalise
from transforms.RandomFlip import RandomFlip
from transforms.RandomNoise import RandomNoise
from transforms.RandomRot import RandomRot
from utils.parser import get_parser_with_args

neptune_logger = NeptuneLogger(
    project="zygisau/train-additional",
    api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyODJmMDUwMS02NGRmLTRiZGQtYWZlNS0xYmI4ZmFiYjZmMzIifQ==",
    log_model_checkpoints=True,
)


if __name__ == "__main__":
    config_parser, _ = get_parser_with_args('config.json')
    opt = config_parser.parse_args()
    neptune_logger.log_hyperparams(opt)

    transform = tr.Compose(
        [Normalise(), RandomFlip(), RandomRot()])
    dataset = OSCDLightning(opt.dataset, opt.batch_size, band_mode='all',
                            transform=transform, num_workers=4)
    # dataloader = dataset.setup('fit')
    # dataloader = dataset.train_dataloader()
    # for I1, I2, mask in dataloader:
    #     # show images 2 rows 3 columns
    #     plt.figure(figsize=(10, 10))
    #     plt.subplot(2, 3, 1)
    #     I1_img = I1[:, [0, 1, 2], :, :].squeeze()
    #     plt.imshow(I1_img.permute(1, 2, 0))
    #     I2_img = I2[:, [0, 1, 2], :, :].squeeze()
    #     plt.subplot(2, 3, 2)
    #     plt.imshow(I2_img.permute(1, 2, 0))
    #     plt.subplot(2, 3, 3)
    #     plt.imshow(mask.squeeze())

    #     TI1, TI2, Tmask = transform((I1.clone(), I2.clone(), mask.clone()))
    #     plt.subplot(2, 3, 4)
    #     TI1_img = TI1[:, [0, 1, 2], :, :].squeeze()
    #     plt.imshow(TI1_img.permute(1, 2, 0))
    #     TI2_img = TI2[:, [0, 1, 2], :, :].squeeze()
    #     plt.subplot(2, 3, 5)
    #     plt.imshow(TI2_img.permute(1, 2, 0))
    #     plt.subplot(2, 3, 6)
    #     plt.imshow(Tmask.squeeze())
    #     plt.show()

    checkpoints = [
        ModelCheckpoint(auto_insert_metric_name=True, monitor='metrics_train_prec',
                        save_top_k=3, save_last=True, every_n_epochs=1),
        ModelCheckpoint(auto_insert_metric_name=True, monitor='metrics_train_acc',
                        save_top_k=3, save_last=True, every_n_epochs=1),
        ModelCheckpoint(auto_insert_metric_name=True, monitor='metrics_valid_prec',
                        save_top_k=3, save_last=True, every_n_epochs=1),
        ModelCheckpoint(auto_insert_metric_name=True, monitor='metrics_valid_acc',
                        save_top_k=3, save_last=True, every_n_epochs=1),
        ModelCheckpoint(auto_insert_metric_name=True, monitor='metrics_train_prec',
                        save_last=True, every_n_train_steps=1000),
        ModelCheckpoint(auto_insert_metric_name=True, monitor='metrics_train_acc',
                        save_last=True, every_n_train_steps=1000),
        ModelCheckpoint(auto_insert_metric_name=True, monitor='metrics_valid_prec',
                        save_last=True, every_n_train_steps=1000),
        ModelCheckpoint(auto_insert_metric_name=True, monitor='metrics_valid_acc',
                        save_last=True, every_n_train_steps=1000)
    ]

    print("=== Best models ===")
    print([c.best_model_path for c in checkpoints])
    print("===================")
    print("\n")

    batch_transform = AppendFeatures(
        opt.feature_model_path, opt.feature_model_checkpoint_path)
    context = Context(log_images=False)
    model = SiamLightning(bands='all', lr=opt.lr, transform=batch_transform,
                          model_checkpoint=opt.siam_checkpoint_path, get_weights=dataset.weights, context=context)
    # model = SiamLightningSigmoid(bands='all', lr=opt.lr, transform=batch_transform,
    #   model_checkpoint=opt.siam_checkpoint_path, get_weights=dataset.weights)

    last_checkpoint = None
    if last_checkpoint is not None:
        checkpoint = torch.load(last_checkpoint)
        weights = {k: v for k,
                   v in checkpoint["state_dict"].items() if k.startswith("model.")}
        # rename all keys to remove the "model." prefix
        weights = {k[6:]: v for k, v in weights.items()}
        model.model.load_state_dict(weights)
    # last_checkpoint_path = "/scratch/lustre/home/zyau5516/source/train-additional/.neptune/None/version_None/checkpoints/"
    # # last checkpoint by modified date time
    # for file in os.listdir(last_checkpoint_path):
    #     if file.endswith(".ckpt"):
    #         if last_checkpoint is None:
    #             last_checkpoint = file
    #         elif os.path.getmtime(file) > os.path.getmtime(last_checkpoint):
    #             last_checkpoint = file
    # if last_checkpoint is not None:
    #     last_checkpoint = last_checkpoint_path + last_checkpoint
    # print("=== Last checkpoint ===")
    # print(last_checkpoint)
    # print("=======================")
    # print("\n")

    trainer = Trainer(logger=neptune_logger, max_epochs=opt.max_epochs, accelerator='gpu', devices=1,
                      callbacks=checkpoints)

    trainer.test(model, dataset)
    trainer.fit(model, dataset)

    trainer.save_checkpoint("last_checkpoint.ckpt")

    context.log_images = True
    context.log_test_roc = True
    trainer.test(model, dataset)
