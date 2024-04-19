import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, F1Score, Precision, Recall
import torchmetrics

from temp.siamunet_diff import SiamUnet_diff


class SiamLightning(L.LightningModule):
    def __init__(self, bands, lr, transform=None):
        super().__init__()
        self.lr = lr
        self.transform = transform
        if bands == 'rgb':
            self.model = SiamUnet_diff(3, 2, 50)
        elif bands == 'all':
            self.model = SiamUnet_diff(13, 2, 50)
        else:
            raise NotImplementedError

        # if model_checkpoints is not None:
        #     if torch.cuda.is_available():
        #         self.model.load_state_dict(
        #             torch.load(model_checkpoints[bands]))
        #     else:
        #         self.model.load_state_dict(torch.load(
        #             model_checkpoints[bands], map_location=torch.device('cpu')))

        self.valid_precision = Precision('binary')
        self.valid_recall = Recall('binary')
        self.valid_f1 = F1Score('binary')
        self.valid_accuracy = Accuracy('binary')

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        image1, image2, y, feat1, feat2 = self.transform(batch)

        y_hat = self.model(image1, image2, feat1, feat2)
        loss = F.nll_loss(input=y_hat, target=y.long())
        self.log("metrics/batch/loss", loss, prog_bar=False)

        y_true = y
        y_pred = y_hat.argmax(axis=1)
        acc = torchmetrics.functional.accuracy(y_true, y_pred, task="binary")
        prec = torchmetrics.functional.precision(y_true, y_pred, task="binary")
        rec = torchmetrics.functional.recall(y_true, y_pred, task="binary")
        self.log("metrics_batch_acc", acc)
        self.log("metrics_batch_prec", prec)
        self.log("metrics_batch_rec", rec)

        return {"loss": loss, "y_true": y_true, "y_pred": y_pred}

    def training_epoch_end(self, outputs):
        loss = np.array([])
        y_true = np.array([])
        y_pred = np.array([])
        for results_dict in outputs:
            loss = np.append(loss, results_dict["loss"].cpu().numpy())
            y_true = np.append(y_true, results_dict["y_true"])
            y_pred = np.append(y_pred, results_dict["y_pred"])
        y_true = torch.tensor(y_true)
        y_pred = torch.tensor(y_pred)
        acc = torchmetrics.functional.accuracy(y_true, y_pred, task="binary")
        self.log("metrics_epoch_loss", loss.mean())
        self.log("metrics_epoch_acc", acc)

    def validation_step(self, batch, batch_idx):
        image1, image2, y, feat1, feat2 = self.transform(batch)
        y_hat = self.model(image1, image2, feat1, feat2)
        loss = F.nll_loss(input=y_hat, target=y.long())

        y_true = y.cpu().detach().numpy()
        y_pred = y_hat.argmax(axis=1).cpu().detach().numpy()

        return {"loss": loss, "y_true": y_true, "y_pred": y_pred}

    def validation_epoch_end(self, outputs):
        loss = np.array([])
        y_true = np.array([])
        y_pred = np.array([])
        for results_dict in outputs:
            loss = np.append(loss, results_dict["loss"].cpu().numpy())
            y_true = np.append(y_true, results_dict["y_true"])
            y_pred = np.append(y_pred, results_dict["y_pred"])
        y_true = torch.tensor(y_true)
        y_pred = torch.tensor(y_pred)
        acc = torchmetrics.functional.accuracy(y_true, y_pred, task="binary")
        prec = torchmetrics.functional.precision(y_true, y_pred, task="binary")
        rec = torchmetrics.functional.recall(y_true, y_pred, task="binary")
        self.log("metrics_val_loss", loss.mean())
        self.log("metrics_val_acc", acc)
        self.log("metrics_val_prec", prec)
        self.log("metrics_val_rec", rec)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), weight_decay=1e-4, lr=self.lr)
