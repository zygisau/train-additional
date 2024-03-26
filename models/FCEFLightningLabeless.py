import lightning as L
import numpy as np
import torch
import torch.nn.functional as F

from models.fresunet import FresUNet
from torchmetrics import JaccardIndex, Precision, Recall, F1Score


def kappa(tp, tn, fp, fn):
    N = tp + tn + fp + fn
    p0 = (tp + tn) / N
    pe = ((tp+fp)*(tp+fn) + (tn+fp)*(tn+fn)) / (N * N)

    return (p0 - pe) / (1 - pe)


class FCEFLightningLabeless(L.LightningModule):
    def __init__(self, model_checkpoints, bands):
        super().__init__()
        if bands == 'rgb':
            self.model = FresUNet(2*3, 2)
        elif bands == 'all':
            self.model = FresUNet(2*13, 2)
        else:
            raise NotImplementedError

        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(model_checkpoints[bands]))
        else:
            self.model.load_state_dict(torch.load(
                model_checkpoints[bands], map_location=torch.device('cpu')))

        self.valid_precision = Precision('binary')
        self.valid_recall = Recall('binary')
        self.valid_f1 = F1Score('binary')
        self.valid_iou = JaccardIndex('binary')

    def forward(self, inputs, target):
        return self.model(inputs, target)

    def test_step(self, batch, batch_idx):
        image1, image2 = batch[0].float(
        ), batch[1].float()
        outputs = self.model(image1, image2)
        return None

    def on_test_end(self) -> None:
        # self.log('acc_epoch', self.valid_precision.compute())
        # self.log('recall_epoch', self.valid_recall.compute())
        # self.log('f1_epoch', self.valid_f1.compute())
        self.valid_precision.reset()
        self.valid_recall.reset()
        self.valid_f1.reset()
        self.valid_iou.reset()
        return super().on_test_end()

    def eval_step(self, output, cm, loss):
        _, predicted = torch.max(output.data, 1)

        self.valid_precision(predicted, cm)
        self.log('test_precision', self.valid_precision)

        self.valid_recall(predicted, cm)
        self.log('test_recall', self.valid_recall)

        self.valid_f1(predicted, cm)
        self.log('test_f1', self.valid_f1)

        self.valid_iou(predicted, cm)
        self.log('test_iou', self.valid_iou)

        # self.tot_loss += loss.data * np.prod(cm.size())
        # self.tot_count += np.prod(cm.size())

        # _, predicted = torch.max(output.data, 1)

        # c = (predicted.int() == cm.data.int())
        # for i in range(c.size(1)):
        #     for j in range(c.size(2)):
        #         l = int(cm.data[0, i, j])
        #         self.class_correct[l] += c[0, i, j]
        #         self.class_total[l] += 1

        # pr = (predicted.int() > 0).cpu().numpy()
        # gt = (cm.data.int() > 0).cpu().numpy()

        # self.tp += np.logical_and(pr, gt).sum()
        # self.tn += np.logical_and(np.logical_not(pr),
        #                           np.logical_not(gt)).sum()
        # self.fp += np.logical_and(pr, np.logical_not(gt)).sum()
        # self.fn += np.logical_and(np.logical_not(pr), gt).sum()

    def training_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass
