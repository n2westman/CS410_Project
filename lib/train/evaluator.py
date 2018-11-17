#__date__ = 6/14/18
#__time__ = 4:07 PM
#__author__ = isminilourentzou

import logging
import lib
import re
from sklearn.metrics import classification_report


class Evaluator(object):
    def __init__(self, model, opt):
        self.model = model
        self.end_epoch  = opt.end_epoch
        self.opt = opt

    def eval(self, data_iter, pred_file=None):
        if(isinstance(self.model, lib.model.CRFTagger)):
            return self.eval_plain(data_iter, pred_file)
        else:
            return self.eval_nn(data_iter, pred_file)

    def eval_plain(self, data_iter, pred_file=None):
        samples, golds = self.model.iter_to_xy(data_iter)
        _, _, preds =  self.model(samples, golds)
        preds = lib.utils.indices2words(preds, self.model.wordrepr.tag_vocab)

        acc, f1, prec, rec = lib.utils.eval_ner(golds, preds)
        if (pred_file):
            temp_samples =[]
            for s in samples:
                new_sample  =[]
                for k in s:
                    new_sample.append(re.sub('word.lower=', '', k[1]))
                temp_samples.append(new_sample)
            samples = temp_samples
            lib.utils.save_predictions(pred_file, preds, golds, samples)
        return -1, acc, f1, prec, rec

    def eval_nn(self, data_iter, pred_file=None):
        self.model.eval()
        total_accuracy, total_f1, total_prec, total_rec, total_loss = 0, 0, 0, 0, 0
        nbatches = len(data_iter)
        preds, golds, samples = [], [], []
        for i, batch in enumerate(data_iter):

            loss, _, pred = self.model(batch)
            loss = loss.item()
            total_loss+=loss

            y_true = lib.utils.indices2words(batch.labels.data.tolist(), self.model.wordrepr.tag_vocab)
            pred = lib.utils.indices2words(pred, self.model.wordrepr.tag_vocab)
            tokens = lib.utils.indices2words(batch.inputs_word.data.tolist(), self.model.wordrepr.word_vocab)

            preds.extend(pred)
            golds.extend(y_true)
            samples.extend(tokens)

        total_accuracy, total_f1, total_prec, total_rec = lib.utils.eval_ner(golds, preds)
        if (pred_file):
            lib.utils.save_predictions(pred_file, preds, golds, samples)
        self.model.train()
        return total_loss/float(nbatches), total_accuracy, total_f1, total_prec, total_rec