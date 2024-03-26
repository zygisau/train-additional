
from lightning import LightningModule
import torch
from torchmetrics.classification import Accuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryJaccardIndex
from models.segmentation import get_segmentation_model
from torch.nn import BCEWithLogitsLoss

import utils.utils as utils
import utils.loss as loss_function


class DinoLightning(LightningModule):

    def __init__(self, backbone, feature_indices, feature_channels, args):
        super().__init__()
        self.model = get_segmentation_model(
            backbone, feature_indices, feature_channels, args.arch)
        if 'BCE' in args.loss_function:
            self.criterion = BCEWithLogitsLoss()
        elif 'dice' in args.loss_function:
            self.criterion = loss_function.dice_bce_loss()

        self.prec = BinaryPrecision(multidim_average='global', threshold=0.5)
        self.rec = BinaryRecall(threshold=0.5)
        self.f1 = BinaryF1Score(threshold=0.5)
        self.iou = BinaryJaccardIndex(threshold=0.5)
        self.args = args

    def forward(self, x1, x2):
        return self.model(x1, x2)

    def training_step(self, batch, batch_idx):
        img_1, img_2, mask, pred, loss, prec, rec, f1, iou = self.shared_step(
            batch)
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/precision', prec, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log('train/recall', rec, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log('train/f1', f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/iou', iou, on_step=False, on_epoch=True, prog_bar=True)
        tensorboard = self.logger.experiment
        global_step = self.trainer.global_step
        tensorboard.add_image('train/img_1', img_1[0], global_step)
        tensorboard.add_image('train/img_2', img_2[0], global_step)
        tensorboard.add_image('train/mask', mask[0], global_step)
        tensorboard.add_image('train/out', (pred[0] >= 0.2)*1, global_step)
        # print((pred[0]>0.5)*1)
        return loss

    def validation_step(self, batch, batch_idx):
        img_1, img_2, mask, pred, loss, prec, rec, f1, iou = self.shared_step(
            batch)
        self.log('val/loss', loss, prog_bar=True)
        self.log('val/precision', prec, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log('val/recall', rec, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log('val/f1', f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/iou', iou, on_step=False, on_epoch=True, prog_bar=True)
        # tensorboard = self.logger.experiment
        # global_step = self.trainer.global_step
        # assert (len(img_1) == len(img_2) == len(mask) == len(pred))
        # for i in range(len(img_1)):
        #     print(str(i), ':', self.cal_f1(pred[i], mask[i]))
        #     tensorboard.add_image('val/'+str(i)+'/img_1',
        #                           img_1[i], global_step)
        #     tensorboard.add_image('val/'+str(i)+'/img_2',
        #                           img_2[i], global_step)
        #     tensorboard.add_image('val/'+str(i)+'/mask', mask[i], global_step)
        #     tensorboard.add_image(
        #         'val/'+str(i)+'/out'+str(self.cal_f1(pred[i], mask[i])), (pred[i] >= 0.2)*1, global_step)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def cal_f1(self, pred, mask):
        f1 = self.f1(pred, mask.long())
        return format(f1, '.4f')

    def shared_step(self, batch):
        image1, image2, mask = batch[0].float(
        ), batch[1].float(), batch[2].long()

        out = self(image1, image2)
        _, out = torch.max(out.data, 1)
        pred = torch.sigmoid(out)

        prec = self.prec(pred, mask.long())
        rec = self.rec(pred, mask.long())
        f1 = self.f1(pred, mask.long())
        iou = self.iou(pred, mask.long())

        if 'BCE' in self.args.loss_function:
            loss = self.criterion(out.float(), mask.float())
        elif 'dice' in self.args.loss_function:
            loss = self.criterion(pred, out, mask)
        return image1, image2, mask, pred, loss, prec, rec, f1, iou

    def configure_optimizers(self):
        params = set(self.model.parameters()).difference(
            self.model.encoder.parameters())
        optimizer = torch.optim.Adam(
            params, lr=self.args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
        return [optimizer], [scheduler]
