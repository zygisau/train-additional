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


class FCEFLightning(L.LightningModule):
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
        # self.tot_loss = 0
        # self.tot_count = 0
        # self.tot_accurate = 0
        # self.n = 2
        # self.class_correct = list(0. for i in range(self.n))
        # self.class_total = list(0. for i in range(self.n))
        # self.class_accuracy = list(0. for i in range(self.n))
        # self.tp = 0
        # self.tn = 0
        # self.fp = 0
        # self.fn = 0

    def forward(self, inputs):
        return self.model(inputs[0].float(), inputs[1].float())

    def test_step(self, batch, batch_idx):
        image1, image2, mask = batch[0].float(
        ), batch[1].float(), batch[2].long()
        outputs = self.model(image1, image2)
        loss = F.nll_loss(input=outputs, target=mask)
        self.eval_step(outputs, mask, loss)
        # self.log_step()
        return loss

    def on_test_end(self) -> None:
        # self.log('acc_epoch', self.valid_precision.compute())
        # self.log('recall_epoch', self.valid_recall.compute())
        # self.log('f1_epoch', self.valid_f1.compute())
        self.valid_precision.reset()
        self.valid_recall.reset()
        self.valid_f1.reset()
        self.valid_iou.reset()
        return super().on_test_end()

    # def log_step(self):
    #     net_loss = self.tot_loss/self.tot_count
    #     net_loss = float(net_loss.cpu().numpy())

    #     net_accuracy = 100 * (self.tp + self.tn)/self.tot_count

    #     self.class_accuracy = list(0. for i in range(self.n))
    #     for i in range(self.n):
    #         self.class_accuracy[i] = 100 * self.class_correct[i] / \
    #             max(self.class_total[i], 0.00001)
    #         self.class_accuracy[i] = float(
    #             self.class_accuracy[i].cpu().numpy())

    #     precision = self.tp / (self.tp + self.fp)
    #     rec = self.tp / (self.tp + self.fn)
    #     dice = 2 * precision * rec / (precision + rec)
    #     precision_nc = self.tn / (self.tn + self.fn)
    #     rec_nc = self.tn / (self.tn + self.fp)

    #     pr_rec = [precision, rec, dice, precision_nc, rec_nc]

    #     k = kappa(self.tp, self.tn, self.fp, self.fn)
    #     self.log_dict({
    #         'net_loss': net_loss,
    #         'net_accuracy': net_accuracy,
    #         'class_accuracy_0': self.class_accuracy[0],
    #         'class_accuracy_1': self.class_accuracy[1],
    #         'precision': precision,
    #         'recall': rec,
    #         'dice': dice,
    #         'kappa': k
    #     })

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
