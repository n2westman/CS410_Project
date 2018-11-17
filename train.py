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

logging.basicConfig(
    filename=os.path.join(
        opt.save_dir,
        opt.al_method +
        'only.log') if opt.logfolder else None,
    level=logging.INFO)
logging.getLogger("data").setLevel('WARNING')
logging.getLogger("model").setLevel('WARNING')
logging.getLogger("train").setLevel('WARNING')
logger = logging.getLogger("train")

if __name__ == '__main__':
    logging.basicConfig(filename='train.log', level=logging.DEBUG)

    dataset = lib.data.Conll_dataset(opt, tag_type='ner', train=True)
    train_iter, validation_iter, test_iter = dataset.batch_iter(batch_size=32)

    for idx, batch in enumerate(train_iter):
        print(batch.labels)
        print(batch.inputs_word)
        exit(0)
