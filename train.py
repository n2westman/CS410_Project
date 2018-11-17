from lib.data import Conll_dataset
import logging

if __name__ == '__main__':
    opt = Conll_dataset.Options('en', 'en')
    logging.basicConfig(filename='train.log',level=logging.DEBUG)

    dataset = Conll_dataset(opt, tag_type='ner', train=True)
    train_iter, validation_iter, test_iter = dataset.batch_iter(batch_size=32)

    for idx, batch in enumerate(train_iter):
        print(batch)
        exit(0)
