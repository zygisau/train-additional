import lightning as L
import matplotlib as mpl
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
from matplotlib import pyplot as plt
from torchmetrics.classification import (BinaryAccuracy, BinaryF1Score,
                                         BinaryJaccardIndex, BinaryPrecision,
                                         BinaryRecall, BinaryROC)

from context import Context
from models.dice_bce_loss import dice_bce_loss
from models.focal_loss import FocalLoss
from temp.siamunet_diff import SiamUnet_diff

mpl.rcParams['figure.dpi'] = 300
mpl.rcParams.update({'font.size': 22})


class SiamLightning(L.LightningModule):
    def __init__(self, bands, lr, transform=None, model_checkpoint=None, get_weights=None, context=Context()):
        super().__init__()
        self.get_weights = get_weights if get_weights is not None else lambda: None
        self.lr = lr
        self.transform = transform
        self.context = context
        if bands == 'rgb':
            self.model = SiamUnet_diff(3, 1, 50)
        elif bands == 'all':
            self.model = SiamUnet_diff(13, 1, 50)
        else:
            raise NotImplementedError

        if model_checkpoint is not None:
            if torch.cuda.is_available():
                checkpoint = torch.load(model_checkpoint)
            else:
                checkpoint = torch.load(
                    model_checkpoint, map_location=torch.device('cpu'))

            model_dict = self.model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in checkpoint.items(
            ) if k in self.model.state_dict() and k not in ['upconv4.weight', 'upconv4.bias', 'conv11d.weight', 'conv11d.bias']}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            self.model.load_state_dict(pretrained_dict, strict=False)

        self.train_precision = BinaryPrecision(
            multidim_average='global', threshold=0.5).to('cuda')
        self.train_recall = BinaryRecall(threshold=0.5).to('cuda')
        self.train_f1 = BinaryF1Score(threshold=0.5).to('cuda')
        self.train_accuracy = BinaryAccuracy(threshold=0.5).to('cuda')
        self.train_iou = BinaryJaccardIndex(threshold=0.5).to('cuda')

        self.valid_precision = BinaryPrecision(
            multidim_average='global', threshold=0.5).to('cuda')
        self.valid_recall = BinaryRecall(threshold=0.5).to('cuda')
        self.valid_f1 = BinaryF1Score(threshold=0.5).to('cuda')
        self.valid_accuracy = BinaryAccuracy(threshold=0.5).to('cuda')
        self.valid_iou = BinaryJaccardIndex(threshold=0.5).to('cuda')

        # self.focal_loss = FocalLoss(focusing_param=3.5, balance_param=0.05)
        self.dice_bce_loss = dice_bce_loss()

    def forward(self, inputs):
        return self.model(inputs)

    def calc_loss(self, y_pred, y):
        bce = F.binary_cross_entropy(input=y_pred, target=y)
        # focal = self.focal_loss(y_pred, y)
        dice = self.dice_bce_loss(y_pred, y)
        loss = (1/3) * bce
        # loss += (1/3) * focal
        loss += (1/3) * dice
        return loss, (bce, 0, dice)

    def log_pred(self, image1, image2, y_pred, y):
        def contrast_stretching(image):
            p2, p98 = np.percentile(image, (2, 98))
            return np.clip((image - p2) * 255.0 / (p98 - p2), 0, 255).astype(np.uint8)

        if not self.context.log_images:
            return

        fig, ax = plt.subplots(1, 4)
        img1 = image1[0, [0, 1, 2], :, :].cpu().numpy().transpose(1, 2, 0)
        img1 = contrast_stretching(img1)
        ax[0].imshow(img1)
        img2 = image2[0, [0, 1, 2], :, :].cpu().numpy().transpose(1, 2, 0)
        img2 = contrast_stretching(img2)
        ax[1].imshow(img2)
        ax[2].imshow((y[0, :, :].cpu().numpy() * 255).astype(np.uint8))
        ax[3].imshow(
            (y_pred[0, :, :].cpu().detach().numpy() * 255).astype(np.uint8))
        # labels
        ax[0].set_title("Image 1")
        ax[1].set_title("Image 2")
        ax[2].set_title("True")
        ax[3].set_title("Pred")
        for a in ax:
            a.axis("off")
        self.logger.experiment["train/pred"].append(fig)
        # clean fig
        plt.close(fig)

    def plot_roc(self, y_pred, y, name="train/roc"):
        assert y_pred.shape == y.shape
        broc = BinaryROC().to('cuda')
        broc.update(y_pred, y)
        fig, ax = broc.plot(score=True)
        mpl.rcParams.update({'font.size': 22})
        fig.set_dpi(300)
        plt.savefig(f"./{name.replace('/', '_')}.pdf", dpi=300)
        plt.savefig(f"./{name.replace('/', '_')}.png", dpi=300)
        self.logger.experiment[name].append(fig)
        plt.close(fig)

    def training_step(self, batch, batch_idx):
        image1, image2, y, feat1, feat2 = self.transform(batch)

        y_pred = self.model(image1, image2, feat1, feat2).squeeze()
        loss, losses = self.calc_loss(y_pred, y)
        self.log("metrics_train_bce", losses[0], on_step=False,
                 on_epoch=True, prog_bar=False)
        self.log("metrics_train_focal", losses[1], on_step=False,
                 on_epoch=True, prog_bar=False)
        self.log("metrics_train_dice", losses[2], on_step=False,
                 on_epoch=True, prog_bar=False)
        # self.log_pred(image1, image2, y_pred, y)
        self.log("metrics_train_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=False)

        y_true = y.long()
        precision = self.train_precision(y_pred, y_true)
        recall = self.train_recall(y_pred, y_true)
        f1 = self.train_f1(y_pred, y_true)
        acc = self.train_accuracy(y_pred, y_true)
        iou = self.valid_iou(y_pred, y_true)

        self.log("metrics_train_prec", precision, on_step=False, on_epoch=True)
        self.log("metrics_train_rec", recall, on_step=False, on_epoch=True)
        self.log("metrics_train_f1", f1, on_step=False, on_epoch=True)
        self.log("metrics_train_acc", acc, on_step=False, on_epoch=True)
        self.log("metrics_train_iou", iou, on_step=False, on_epoch=True)

        # return {"loss": loss, "y_true": y_true, "y_pred": y_pred}

    # def training_epoch_end(self, outputs):
    #     loss = np.array([])
    #     y_true = np.array([])
    #     y_pred = np.array([])
    #     for results_dict in outputs:
    #         loss = np.append(loss, results_dict["loss"].numpy())
    #         y_true = np.append(y_true, results_dict["y_true"])
    #         y_pred = np.append(y_pred, results_dict["y_pred"])
    #     y_true = torch.tensor(y_true)
    #     y_pred = torch.tensor(y_pred)
    #     acc = torchmetrics.functional.accuracy(y_true, y_pred, task="binary")
    #     self.log("metrics_epoch_loss", loss.mean())
    #     self.log("metrics_epoch_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        image1, image2, y, feat1, feat2 = self.transform(batch)

        y_pred = self.model(image1, image2, feat1, feat2).squeeze()
        loss, losses = self.calc_loss(y_pred, y)
        self.log("metrics_valid_bce", losses[0], on_step=False,
                 on_epoch=True, prog_bar=False)
        self.log("metrics_valid_focal", losses[1], on_step=False,
                 on_epoch=True, prog_bar=False)
        self.log("metrics_valid_dice", losses[2], on_step=False,
                 on_epoch=True, prog_bar=False)
        self.log("metrics_valid_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=False)

        y_true = y.long()
        precision = self.train_precision(y_pred, y_true)
        recall = self.valid_recall(y_pred, y_true)
        f1 = self.valid_f1(y_pred, y_true)
        accuracy = self.valid_accuracy(y_pred, y_true)
        iou = self.valid_iou(y_pred, y_true)

        self.log("metrics_valid_prec", precision, on_step=False, on_epoch=True)
        self.log("metrics_valid_rec", recall, on_step=False, on_epoch=True)
        self.log("metrics_valid_f1", f1, on_step=False, on_epoch=True)
        self.log("metrics_valid_acc", accuracy, on_step=False, on_epoch=True)
        self.log("metrics_valid_iou", iou, on_step=False, on_epoch=True)

        return {"loss": loss, "y_true": y_true, "y_pred": y_pred}
        # return loss

    # def validation_epoch_end(self, outputs):
    #     last_output = outputs[-1]
    #     y_true, y_pred = last_output["y_true"], last_output["y_pred"]
    #     self.plot_roc(y_pred, y_true, 'valid/roc')

    def test_step(self, batch, batch_idx):
        image1, image2, y, feat1, feat2 = self.transform(batch)

        y_pred = self.model(image1, image2, feat1, feat2).squeeze()
        loss, losses = self.calc_loss(y_pred, y)
        self.log("metrics_test_bce", losses[0], on_step=False,
                 on_epoch=True, prog_bar=False)
        self.log("metrics_test_focal", losses[1], on_step=False,
                 on_epoch=True, prog_bar=False)
        self.log("metrics_test_dice", losses[2], on_step=False,
                 on_epoch=True, prog_bar=False)
        self.log("metrics_test_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=False)

        y_true = y.long()
        precision = self.train_precision(y_pred, y_true)
        recall = self.valid_recall(y_pred, y_true)
        f1 = self.valid_f1(y_pred, y_true)
        accuracy = self.valid_accuracy(y_pred, y_true)
        iou = self.valid_iou(y_pred, y_true)

        self.log("metrics_test_prec", precision, on_step=False, on_epoch=True)
        self.log("metrics_test_rec", recall, on_step=False, on_epoch=True)
        self.log("metrics_test_f1", f1, on_step=False, on_epoch=True)
        self.log("metrics_test_acc", accuracy, on_step=False, on_epoch=True)
        self.log("metrics_test_iou", iou, on_step=False, on_epoch=True)
        self.log_pred(image1, image2, y_pred, y)

        return {"loss": loss, "y_true": y_true, "y_pred": y_pred}
        # return loss

    def test_epoch_end(self, outputs):
        if not self.context.log_test_roc:
            return

        # concat all torch tensors
        y_true = []
        y_pred = []
        for results_dict in outputs:
            y_true.append(results_dict["y_true"].cpu().detach().numpy())
            y_pred.append(results_dict["y_pred"].cpu().detach().numpy())

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)

        y_true = torch.tensor(y_true)
        y_pred = torch.tensor(y_pred)
        self.plot_roc(y_pred, y_true.long(), 'test/roc')

    # def validation_epoch_end(self, outputs):
    #     loss = np.array([])
    #     y_true = np.array([])
    #     y_pred = np.array([])
    #     for results_dict in outputs:
    #         loss = np.append(loss, results_dict["loss"].numpy())
    #         y_true = np.append(y_true, results_dict["y_true"])
    #         y_pred = np.append(y_pred, results_dict["y_pred"])
    #     y_true = torch.tensor(y_true)
    #     y_pred = torch.tensor(y_pred)
    #     acc = torchmetrics.functional.accuracy(y_true, y_pred, task="binary")
    #     prec = torchmetrics.functional.precision(y_true, y_pred, task="binary")
    #     rec = torchmetrics.functional.recall(y_true, y_pred, task="binary")
    #     self.log("metrics_val_loss", loss.mean())
    #     self.log("metrics_val_acc", acc)
    #     self.log("metrics_val_prec", prec)
    #     self.log("metrics_val_rec", rec)
        # return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=1e-4)

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
        return [optimizer], [scheduler]
        # return optimizer
        # return torch.optim.Adam(self.parameters(), weight_decay=1e-4, lr=self.lr)
