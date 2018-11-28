# -*- coding: utf-8 -*-

from itertools import count
import os
import numpy as np
from config import opt, additional_args
import logging
import time
import lib

start_time = time.time()
additional_args(opt)
N_SAMPLES = opt.nsamples

logging.basicConfig(filename='train.log', level=logging.INFO)
logger = logging.getLogger("train")

def print_samples(n, dataset):
    train_iter, _, _ = dataset.batch_iter(batch_size=n)

    for i, batch in enumerate(train_iter):
        sentences = [' '.join(l) for l in lib.utils.indices2words(batch.inputs_word, dataset.vocabs[0], remove_pad=True)]
        labels = [' '.join(l) for l in lib.utils.indices2words(batch.labels, dataset.vocabs[3], remove_pad=True)]
        for j, sentence in enumerate(sentences):
            print('Example %d: %s\nLabel %d: %s' % (j+1, sentence, j+1, labels[j]))
        break

def main():
    logger.info(opt)

    dataset = lib.data.Conll_dataset(opt, tag_type='ner', train=True)
    train_iter, validation_iter, test_iter = dataset.batch_iter(batch_size=2)

    wordrepr =  lib.train.build_wordrepr(opt, dataset.vocabs)

    train_iter, validation_iter, test_iter = dataset.batch_iter(opt.batch_size)
    model, optim = lib.train.create_model(opt, wordrepr)
    trainer = lib.train.Trainer(model, train_iter, validation_iter, optim, opt)

    _, _, _, _, acc, f1, prec, rec = trainer.train(opt.start_epoch, opt.end_epoch)[-1]
    logger.info('First accuracy: %.3f, f1: %.3f, prec: %.3f, rec: %.3f' % (acc, f1, prec, rec))

    logger.info('Test finished.')
    logger.info("--- %s seconds ---" % (time.time() - start_time))

    # print_samples(10, dataset)

if __name__ == '__main__':
    main()
