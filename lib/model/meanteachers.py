# https://github.com/lyakaap/VAT-pytorch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def _sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def _get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 100. * _sigmoid_rampup(epoch, 3)

def _loss(pred, pred_hat):
    return F.mse_loss(pred, pred_hat) / pred.shape[0]

class MeanTeachers(nn.Module):

    def __init__(self, opt):
        """VAT loss
        :param opt: configuation options as defined in config.py
        """
        super(MeanTeachers, self).__init__()
        self.opt = opt


    def forward(self, model, ema_model, batch, epoch):
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

        consistency_loss = _loss(pred, pred_hat)
        return _get_current_consistency_weight(epoch) * consistency_loss
