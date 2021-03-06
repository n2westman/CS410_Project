# https://github.com/lyakaap/VAT-pytorch

import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):

    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d

def _loss(pred, pred_hat):
    return F.mse_loss(pred, pred_hat) / pred.shape[0]

class VATLoss(nn.Module):

    def __init__(self, xi=10.0, eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, batch):
        with torch.no_grad():
            pred = model.predict(batch)

        # prepare random unit tensor
        x = model.get_intermediate_layer(0, batch)
        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                emb_noise = self.xi * d
                pred_hat = model.predict(batch, emb_noise=emb_noise)
                adv_distance = _loss(pred, pred_hat)
                adv_distance.backward(retain_graph=True)
                d = _l2_normalize(d.grad)
                model.zero_grad()

            # calc LDS
            r_adv = d * self.eps
            pred_hat = model.predict(batch, emb_noise=r_adv)
            lds = _loss(pred, pred_hat)

        return lds
