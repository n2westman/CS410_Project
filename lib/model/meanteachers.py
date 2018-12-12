# https://github.com/lyakaap/VAT-pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def _loss(pred, pred_hat):
    return F.mse_loss(pred, pred_hat) / pred.shape[0]

class MeanTeachers(nn.Module):

    def __init__(self, opt):
        """VAT loss
        :param opt: configuation options as defined in config.py
        """
        super(MeanTeachers, self).__init__()
        self.opt = opt


    def forward(self, model, ema_model, batch):
        pred = model.predict(batch)

        with torch.no_grad():
            pred_hat = ema_model.predict(batch)

        if False:
            with torch.no_grad():
                # Visualize an example to see that
                # the models predict similar results.
                print("======== Test =======")
                print(F.softmax(pred, dim=2)[0][0])
                print(F.softmax(pred_hat, dim=2)[0][0])

        consistency_loss = 100. * _loss(pred, pred_hat)
        return consistency_loss
