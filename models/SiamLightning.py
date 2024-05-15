import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryJaccardIndex
import torchmetrics

from models.focal_loss import FocalLoss
from temp.siamunet_diff import SiamUnet_diff


class SiamLightning(L.LightningModule):
    def __init__(self, bands, lr, transform=None, model_checkpoint=None, get_weights=None):
        super().__init__()
        self.get_weights = get_weights if get_weights is not None else lambda: None
        self.lr = lr
        self.transform = transform
        if bands == 'rgb':
            self.model = SiamUnet_diff(3, 2, 50)
        elif bands == 'all':
            self.model = SiamUnet_diff(13, 2, 50)
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
            ) if k in self.model.state_dict() and k not in ['upconv4.weight', 'upconv4.bias']}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            self.model.load_state_dict(pretrained_dict, strict=False)
        
        self.train_precision = BinaryPrecision(multidim_average='global', threshold=0.5).to('cuda')
        self.train_recall = BinaryRecall(threshold=0.5).to('cuda')
        self.train_f1 = BinaryF1Score(threshold=0.5).to('cuda')
        self.train_accuracy = BinaryAccuracy(threshold=0.5).to('cuda')
        self.train_iou = BinaryJaccardIndex(threshold=0.5).to('cuda')

        self.valid_precision = BinaryPrecision(multidim_average='global', threshold=0.5).to('cuda')
        self.valid_recall = BinaryRecall(threshold=0.5).to('cuda')
        self.valid_f1 = BinaryF1Score(threshold=0.5).to('cuda')
        self.valid_accuracy = BinaryAccuracy(threshold=0.5).to('cuda')
        self.valid_iou = BinaryJaccardIndex(threshold=0.5).to('cuda')

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        image1, image2, y, feat1, feat2 = self.transform(batch)

        y_hat = self.model(image1, image2, feat1, feat2)
        loss = F.nll_loss(input=y_hat, target=y.long(),
                          weight=self.get_weights())
        self.log("metrics_train_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=False)

        y_true = y.long()
        _, y_pred = torch.max(y_hat.data, 1)
        y_pred = torch.sigmoid(y_pred)
        precision = self.train_precision(y_pred, y_true)
        recall = self.train_recall(y_pred, y_true)
        f1 = self.train_f1(y_pred, y_true)
        acc = self.train_accuracy(y_pred, y_true)
        iou = self.valid_iou(y_pred, y_true)

        self.log("metrics_train_prec", precision, on_step=True, on_epoch=True)
        self.log("metrics_train_rec", recall, on_step=True, on_epoch=True)
        self.log("metrics_train_f1", f1, on_step=True, on_epoch=True)
        self.log("metrics_train_acc", acc, on_step=True, on_epoch=True)
        self.log("metrics_train_iou", iou, on_step=True, on_epoch=True)

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
        y_hat = self.model(image1, image2, feat1, feat2)
        loss = F.nll_loss(input=y_hat, target=y.long(),
                          weight=self.get_weights())
        self.log("metrics_valid_loss", loss, on_step=False,
                 on_epoch=True, prog_bar=False)

        y_true = y.long()
        _, y_pred = torch.max(y_hat.data, 1)
        y_pred = torch.sigmoid(y_pred)
        precision = self.valid_precision(y_pred, y_true)
        recall = self.valid_recall(y_pred, y_true)
        f1 = self.valid_f1(y_pred, y_true)
        accuracy = self.valid_accuracy(y_pred, y_true)
        iou = self.valid_iou(y_pred, y_true)

        self.log("metrics_valid_prec", precision, on_step=False, on_epoch=True)
        self.log("metrics_valid_rec", recall, on_step=False, on_epoch=True)
        self.log("metrics_valid_f1", f1, on_step=False, on_epoch=True)
        self.log("metrics_valid_acc", accuracy, on_step=False, on_epoch=True)
        self.log("metrics_valid_iou", iou, on_step=False, on_epoch=True)

        # return {"loss": loss, "y_true": y_true, "y_pred": y_pred}
        return loss
    
    def test_step(self, batch, batch_idx):
        image1, image2, y, feat1, feat2 = self.transform(batch)
        y_hat = self.model(image1, image2, feat1, feat2)
        loss = F.nll_loss(input=y_hat, target=y.long(),
                          weight=self.get_weights())
        self.log("metrics_test_loss", loss, on_step=False,
                 on_epoch=True, prog_bar=False)

        y_true = y.long()
        _, y_pred = torch.max(y_hat.data, 1)
        y_pred = torch.sigmoid(y_pred)
        precision = self.valid_precision(y_pred, y_true)
        recall = self.valid_recall(y_pred, y_true)
        f1 = self.valid_f1(y_pred, y_true)
        accuracy = self.valid_accuracy(y_pred, y_true)
        iou = self.valid_iou.to('cpu')(y_pred, y_true)

        self.log("metrics_test_prec", precision, on_step=False, on_epoch=True)
        self.log("metrics_test_rec", recall, on_step=False, on_epoch=True)
        self.log("metrics_test_f1", f1, on_step=False, on_epoch=True)
        self.log("metrics_test_acc", accuracy, on_step=False, on_epoch=True)
        self.log("metrics_test_iou", iou, on_step=False, on_epoch=True)

        # return {"loss": loss, "y_true": y_true, "y_pred": y_pred}
        return loss

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
        # TODO: try to turn scheduler off -> No need, scheduler is orking every epoch
        # TODO: try to test best checkpoint with test data
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
        return [optimizer], [scheduler]
        # return optimizer
        # return torch.optim.Adam(self.parameters(), weight_decay=1e-4, lr=self.lr)
