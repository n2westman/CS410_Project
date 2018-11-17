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

def main():
    logger.info(opt)

    dataset = lib.data.Conll_dataset(opt, tag_type='ner', train=True)
    train_iter, validation_iter, test_iter = dataset.batch_iter(batch_size=32)


    wordrepr =  lib.train.build_wordrepr(opt, dataset.vocabs)
    logger.info('Begin training active learning policy..')

    # for idx, batch in enumerate(train_iter):
    #     print(batch.labels)
    #     print(batch.inputs_word)
    #     exit(0)

if __name__ == '__main__':
    main()
