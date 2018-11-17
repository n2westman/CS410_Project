#__date__ = 6/14/18
#__time__ = 4:09 PM
#__author__ = isminilourentzou

import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import logging

class Optim(object):

    def _makeOptimizer(self):
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def __init__(self, params, method, lr, max_grad_norm, lr_decay=1, start_decay_after=None):
        self.params = list(params)  # careful: params may be a generator
        self.last_loss = None
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_after = start_decay_after

        self._makeOptimizer()

    def step(self):
        """"# Compute gradients norm.
        #print "Enter step"
        grad_norm = 0
        for param in self.params:
            #print "param", param
            #print "param.grad", param.grad
            #print "param.grad.data.norm()", param.grad.data.norm()
            grad_norm += math.pow(param.grad.data.norm(), 2)
        grad_norm = math.sqrt(grad_norm)
        shrinkage = self.max_grad_norm / grad_norm
        for param in self.params:
            if shrinkage < 1:
                param.grad.data.mul_(shrinkage)"""
        clip_grad_norm_(self.params, self.max_grad_norm)
        self.optimizer.step()

    def set_lr(self, lr):
        self.lr = lr
        self.optimizer.param_groups[0]["lr"] = lr

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, d):
        return self.optimizer.load_state_dict(d)

    def update_lr(self, loss, epoch):
        if self.start_decay_after is not None and epoch >= self.start_decay_after:
            if self.last_loss is not None and loss > self.last_loss:
                logging.info("Decaying learning rate from {} to {}".format(self.lr, self.lr * self.lr_decay))
                self.set_lr(self.lr * self.lr_decay)
        self.last_loss = loss