#__date__ = 6/14/18
#__time__ = 4:09 PM
#__author__ = isminilourentzou

import datetime
import os
import time
import numpy as np
import torch
from torchtext.data.example import Example
from torch import nn
from collections import Counter
from torch.utils.data.sampler import WeightedRandomSampler
# TODO: The sklearn scores have a field called 'average'.
# Whoever uses the code should consider changing the default choice
# More info: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

import lib
import logging

logger = logging.getLogger("train")

class Trainer(object):
    def __init__(self, model, train_iter, eval_iter, optim, opt):

        self.model = model
        self.train_iter = train_iter
        self.eval_iter = eval_iter
        self.evaluator = lib.train.Evaluator(model, opt)
        self.optim = optim
        self.opt = opt
        self.plain = opt.plainCRF # isinstance(self.model, lib.model.CRFTagger)

        if (opt.model_type == "VAT"):
            self.VATLoss = lib.model.VATLoss()

    def train(self, start_epoch, end_epoch, save_model=None, start_time=None):
        if(self.plain): start_epoch = end_epoch = 0
        self.start_time = time.time() if start_time is None else start_time
        stats=[]
        for epoch in range(start_epoch, end_epoch + 1):
            logger.info('* Training %s epoch *' % epoch)
            if(self.plain):
                self.model.train(self.train_iter)
                loss, acc, f1, prec, rec = self.evaluator.eval(self.train_iter)
            else:
                logger.info("Model optim lr: %g" % self.optim.lr)
                loss, acc, f1, prec, rec = self.train_epoch(epoch)

            logger.info('Train loss: %.3f, accuracy: %.3f, f1: %.3f, prec: %.3f, rec: %.3f' % (loss,  acc, f1, prec, rec))
            save_pred = os.path.join(self.opt.save_dir, "model_%d.txt" % epoch)
            loss, acc, f1, prec, rec = self.evaluator.eval(self.eval_iter, pred_file=save_pred if save_model else None)
            stats.append([epoch, self.opt.al_method if self.opt.al_method else "-", len(self.train_iter), loss, acc, f1, prec, rec])
            logger.info('Validation loss: %.3f, accuracy: %.3f, f1: %.3f, prec: %.3f, rec: %.3f' % (loss,  acc, f1, prec, rec))
            if not self.plain: self.optim.update_lr(loss, epoch)
            if(save_model):
                self.save(save_model)
        return stats

    def save(self, save_model):
        model_name = os.path.join(self.opt.save_dir, save_model)
        if(self.plain):
            os.rename(self.model.model_file, model_name)
            self.model.tagger.open(model_name)
        else:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optim_state_dict': self.optim.state_dict(),
                'opt': self.opt
            }
            torch.save(checkpoint, model_name)
        logger.info("Save model as %s" % model_name)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss, total_accuracy, report_loss, report_accuracy = 0, 0, 0, 0
        total_prec, total_rec, report_prec, report_rec, total_f1, report_f1 = 0, 0, 0, 0, 0, 0

        nbatches = len(self.train_iter)
        for i, batch in enumerate(self.train_iter):
            self.model.zero_grad()
            loss, scores, pred = self.model(batch)

            if (self.VATLoss):
                lds = self.VATLoss(self.model, batch)
                loss = loss + self.opt.alpha * lds

            loss.backward()
            loss = loss.item()
            self.optim.step()

            y_true = lib.utils.indices2words(batch.labels.data.tolist(), self.model.wordrepr.tag_vocab)
            pred = lib.utils.indices2words(pred, self.model.wordrepr.tag_vocab)

            accuracy, f1, prec, rec = lib.utils.eval_ner(y_true, pred)
            total_loss += loss
            report_loss += loss
            total_accuracy += accuracy
            report_accuracy += accuracy
            total_f1 += f1
            report_f1 += f1
            total_prec += prec
            report_prec += prec
            total_rec += rec
            report_rec += rec

            if (i + 1) % self.opt.log_interval == 0:
                logger.info("""Epoch %3d, %6d/%d batches; loss:%.2f; accuracy:%.2f; f1:%.2f; prec:%.2f; rec:%.2f; %s elapsed""" %
                      (epoch, (i + 1), nbatches, report_loss, report_accuracy, report_f1, report_prec, report_rec,
                       str(datetime.timedelta(seconds=int(time.time() - self.start_time)))))
                report_loss = report_accuracy = report_f1 = report_prec = report_rec = 0
        if(nbatches==0): return 0, 0, 0, 0, 0
        return total_loss/float(nbatches), total_accuracy/float(nbatches), total_f1/float(nbatches), total_prec/float(nbatches), total_rec/float(nbatches)
