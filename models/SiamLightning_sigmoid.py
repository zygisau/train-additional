import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, F1Score, Precision, Recall
import torchmetrics

from models.dice_bce_loss import dice_bce_loss
from models.focal_loss import FocalLoss
from temp.siamunet_diff import SiamUnet_diff


class SiamLightningSigmoid(L.LightningModule):
    def __init__(self, bands, lr, transform=None, model_checkpoint=None, get_weights=None):
        super().__init__()
        self.lr = lr
        self.transform = transform
        self.get_weights = get_weights
        if bands == 'rgb':
            self.model = SiamUnet_diff(3, 2, 50, apply_softmax=False)
        elif bands == 'all':
            self.model = SiamUnet_diff(13, 2, 50, apply_softmax=False)
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

        self.train_precision = Precision('binary')
        self.train_recall = Recall('binary')
        self.train_f1 = F1Score('binary')
        self.train_accuracy = Accuracy('binary')

        self.valid_precision = Precision('binary')
        self.valid_recall = Recall('binary')
        self.valid_f1 = F1Score('binary')
        self.valid_accuracy = Accuracy('binary')

    # def setup(self, stage):
    #     self.criterion = F.binary_cross_entropy(self.get_weights())

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        image1, image2, y_true, feat1, feat2 = self.transform(batch)

        y_out = self.model(image1, image2, feat1, feat2)
        y_pred = torch.sigmoid(y_out)[:,1,:,:]

        # TODO: BCE loss oportunity
        print(y_pred.shape, y_true.shape, self.get_weights())
        loss = F.binary_cross_entropy(y_pred, y_true, self.get_weights())
        self.log("metrics_train_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=False)

        # TODO: Move cpu conversion here
        y_pred, y_true = y_pred.cpu()[:,1,:,:], y_true.cpu()
        precision = self.train_precision(y_pred, y_true)
        recall = self.train_recall(y_pred, y_true)
        f1 = self.train_f1(y_pred, y_true)
        acc = self.train_accuracy(y_pred, y_true)

        self.log("metrics_train_prec", precision, on_step=True, on_epoch=True)
        self.log("metrics_train_rec", recall, on_step=True, on_epoch=True)
        self.log("metrics_train_f1", f1, on_step=True, on_epoch=True)
        self.log("metrics_train_acc", acc, on_step=True, on_epoch=True)

        # return {"loss": loss, "y_true": y_true, "y_pred": y_pred}

    # def training_epoch_end(self, outputs):
    #     loss = np.array([])
    #     y_true = np.array([])
    #     y_pred = np.array([])
    #     for results_dict in outputs:
    #         loss = np.append(loss, results_dict["loss"].cpu().numpy())
    #         y_true = np.append(y_true, results_dict["y_true"])
    #         y_pred = np.append(y_pred, results_dict["y_pred"])
    #     y_true = torch.tensor(y_true)
    #     y_pred = torch.tensor(y_pred)
    #     acc = torchmetrics.functional.accuracy(y_true, y_pred, task="binary")
    #     self.log("metrics_epoch_loss", loss.mean())
    #     self.log("metrics_epoch_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        image1, image2, y_true, feat1, feat2 = self.transform(batch)

        y_out = self.model(image1, image2, feat1, feat2)
        y_pred = torch.sigmoid(y_out)[:,1,:,:]

        # TODO: BCE loss oportunity
        print(y_pred.shape, y_true.shape, self.get_weights())
        loss = F.binary_cross_entropy(y_pred, y_true, self.get_weights())
        self.log("metrics_valid_loss", loss, on_step=False,
                 on_epoch=True, prog_bar=False)

        y_pred, y_true = y_pred.cpu()[:,1,:,:], y_true.cpu()
        precision = self.valid_precision(y_pred, y_true)
        recall = self.valid_recall(y_pred, y_true)
        f1 = self.valid_f1(y_pred, y_true)
        accuracy = self.valid_accuracy(y_pred, y_true)
        self.log("metrics_valid_prec", precision, on_step=False, on_epoch=True)
        self.log("metrics_valid_rec", recall, on_step=False, on_epoch=True)
        self.log("metrics_valid_f1", f1, on_step=False, on_epoch=True)
        self.log("metrics_valid_acc", accuracy, on_step=False, on_epoch=True)

        # return {"loss": loss, "y_true": y_true, "y_pred": y_pred}
        return loss

    # def validation_epoch_end(self, outputs):
    #     loss = np.array([])
    #     y_true = np.array([])
    #     y_pred = np.array([])
    #     for results_dict in outputs:
    #         loss = np.append(loss, results_dict["loss"].cpu().numpy())
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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
        return [optimizer]
        # return torch.optim.Adam(self.parameters(), weight_decay=1e-4, lr=self.lr)
