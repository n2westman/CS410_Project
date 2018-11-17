from data import Conll_dataset;

if __name__ == '__main__':
    dataset = data.Conll_dataset(opt, tag_type='ner', train=False)
    train_iter, validation_iter, test_iter = dataset.batch_iter(batch_size=32)
