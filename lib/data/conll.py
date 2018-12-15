from .crosslingual_vectors import Crosslingual
from torchtext import data
from .NERDataset import NERDataset
from torchtext.datasets import SequenceTaggingDataset
import logging
import numpy as np
import torch
import math

DATA_RELATIVE_PATH = 'data'

logger = logging.getLogger("data")
# predefine a label_set: PER - 1, LOC - 2, ORG - 3, MISC - 4, O - 5
labels_map = {
    'B-ORG': 'ORG',
    'O': 'O',
    'B-MISC': 'MISC',
    'B-PER': 'PER',
    'I-PER': 'PER',
    'B-LOC': 'LOC',
    'I-ORG': 'ORG',
    'I-MISC': 'MISC',
    'I-LOC': 'LOC'}

caseLookup = {
    'numeric': 0,
    'allLower': 1,
    'allUpper': 2,
    'initialUpper': 3,
    'other': 4,
    'mainly_numeric': 5,
    'contains_digit': 6}

mapping_files = {
    'en.train': DATA_RELATIVE_PATH + '/conll2003/eng.train.txt',
    'en.testa': DATA_RELATIVE_PATH + '/conll2003/eng.testa.txt',
    'en.testb': DATA_RELATIVE_PATH + '/conll2003/eng.testb.txt',
    'de.train': DATA_RELATIVE_PATH + '/conll2003/deu.train.txt',
    'de.testa': DATA_RELATIVE_PATH + '/conll2003/deu.testa.txt',
    'de.testb': DATA_RELATIVE_PATH + '/conll2003/deu.testb.txt',
    'es.train': DATA_RELATIVE_PATH + '/conll2002/esp.train.txt',
    'es.testa': DATA_RELATIVE_PATH + '/conll2002/esp.testa.txt',
    'es.testb': DATA_RELATIVE_PATH + '/conll2002/esp.testb.txt',
    'nl.train': DATA_RELATIVE_PATH + '/conll2002/ned.train.txt',
    'nl.testa': DATA_RELATIVE_PATH + '/conll2002/ned.testa.txt',
    'nl.testb': DATA_RELATIVE_PATH + '/conll2002/ned.testb.txt',
    'fifty_nine.cca.normalized': DATA_RELATIVE_PATH + '/fifty_nine.cca.normalized',
    'cadec': DATA_RELATIVE_PATH + '/cadec/cadec.conll'}

class Conll_dataset():

    def __init__(self, opt, train=True, tag_type='ner'):
        self.opt = opt
        opt.lang = opt.train if train else opt.test
        if(opt.lang.lower() == 'cadec'):
            inputs_word, inputs_char, inputs_case, labels = self.cadec(
                opt, tag_type=tag_type)
        else:
            inputs_word, inputs_char, inputs_case, labels = self.conll(
                opt, tag_type=tag_type)
        self.check_ids(self.train)
        self.check_ids(self.val)
        self.check_ids(self.test)
        # Build vocab
        inputs_char.build_vocab(
            self.train.inputs_char,
            self.val.inputs_char,
            self.test.inputs_char,
            max_size=opt.maxcharvocab)
        inputs_case.build_vocab(
            self.train.inputs_case,
            self.val.inputs_case,
            self.test.inputs_case)
        inputs_word.build_vocab(self.train.inputs_word, self.val.inputs_word, self.test.inputs_word, max_size=opt.maxvocab,
                                # vectors ="fasttext.en.300d")
                                vectors=[Crosslingual(mapping_files['fifty_nine.cca.normalized'])] if opt.pre_embs else None)
        labels.build_vocab(self.train.labels)
        self.vocabs = inputs_word.vocab, inputs_char.vocab, inputs_case.vocab, labels.vocab
        self.count_new, self.train_unlabeled = 0, []
        self.gpu = opt.gpu
        self.labeled = opt.labeled
        # Keep for reseting
        self.keep_duplicates()

        if(opt.labeled != -1):
            # Create unlabeled dataset
            ratio = (opt.labeled * 1.) / len(self.train.examples)
            if ratio != 0:
                self.train, self.train_unlabeled = self.train.split(ratio)
            else:
                self.train_unlabeled = data.Dataset(
                    examples=self.train.examples, fields=self.fields)
                self.train.examples = []
        if(opt.budget is None):
            opt.budget = len(self.train_unlabeled)
        logger.info('Train size: %d' % (len(self.train)))
        logger.info('Validation size: %d' % (len(self.val)))
        logger.info('Test size: %d' % (len(self.test)))
        logger.info('Unlabeled size: %d' % (len(self.train_unlabeled)))
        logger.info('Input word vocab size:%d' % (len(inputs_word.vocab)))
        logger.info('Input char vocab size:%d' % (len(inputs_char.vocab)))
        logger.info('Input case vocab size:%d' % (len(inputs_case.vocab)))
        logger.info('Tagset size: %d' % (len(labels.vocab)))
        logger.info('Tag set:[{}]'.format(','.join(labels.vocab.itos)))
        logger.info('----------------------------')

    def conll(self, opt, tag_type='ner'):
        """
           conll2003: Conll 2003 (Parser only. You must place the files)
           Extract Conll2003 dataset using torchtext. Applies GloVe 6B.200d and Char N-gram
           pretrained vectors. Also sets up per word character Field
           tag_type: Type of tag to pick as task [pos, chunk, ner]
        """
        logger.info(
            '---------- CONLL 2003 %s lang = %s ---------' %
            (tag_type, opt.lang))
        train_file = mapping_files['.'.join([opt.lang, 'train'])]
        dev_file = mapping_files['.'.join([opt.lang, 'testa'])]
        test_file = mapping_files['.'.join([opt.lang, 'testb'])]
        encoding = 'utf8' if opt.lang == "en" else 'latin-1'

        # Setup fields with batch dimension first
        inputs_word = data.Field(
            batch_first=True,
            fix_length=opt.maxlen,
            lower=opt.lower,
            preprocessing=data.Pipeline(
                lambda w: '0' if opt.convert_digits and w.isdigit() else w))

        inputs_char_nesting = data.Field(
            tokenize=list, batch_first=True, fix_length=opt.maxlen)
        inputs_char = data.NestedField(inputs_char_nesting)

        inputs_case = data.Field(
            batch_first=True,
            fix_length=opt.maxlen,
            preprocessing=data.Pipeline(
                lambda w: self.getCasing(w)))

        labels = data.Field(batch_first=True, unk_token=None, fix_length=opt.maxlen,  # pad_token=None,
                            preprocessing=data.Pipeline(lambda w: labels_map[w]))

        id = data.Field(batch_first=True, use_vocab=False)

        if(opt.lang == "en"):
            self.fields = ([(('inputs_word',
                              'inputs_char',
                              'inputs_case'),
                             (inputs_word,
                              inputs_char,
                              inputs_case))] + [('labels',
                                                 labels) if label == tag_type else (None,
                                                                                    None) for label in ['pos',
                                                                                                        'chunk',
                                                                                                        'ner']] + [('id',
                                                                                                                    id)])

        elif(opt.lang == "de"):
            self.fields = ([(('inputs_word',
                              'inputs_char',
                              'inputs_case'),
                             (inputs_word,
                              inputs_char,
                              inputs_case))] + [('idk',
                                                 None)] + [('labels',
                                                            labels) if label == tag_type else (None,
                                                                                               None) for label in ['pos',
                                                                                                                   'chunk',
                                                                                                                   'ner']] + [('id',
                                                                                                                               id)])

        elif(opt.lang == "nl"):
            self.fields = ([(('inputs_word',
                              'inputs_char',
                              'inputs_case'),
                             (inputs_word,
                              inputs_char,
                              inputs_case))] + [('labels',
                                                 labels) if label == tag_type else (None,
                                                                                    None) for label in ['pos',
                                                                                                        'ner']] + [('id',
                                                                                                                    id)])
        else:
            self.fields = ([(('inputs_word',
                              'inputs_char',
                              'inputs_case'),
                             (inputs_word,
                              inputs_char,
                              inputs_case))] + [('labels',
                                                 labels) if label == tag_type else (None,
                                                                                    None) for label in ['ner']] + [('id',
                                                                                                                    id)])

        # Load the data
        self.train, self.val, self.test = NERDataset.splits(
            path='.',
            train=train_file,
            validation=dev_file,
            test=test_file,
            separator=' ',
            encoding=encoding,
            fields=tuple(self.fields))
        return inputs_word, inputs_char, inputs_case, labels

    def cadec(self, opt, tag_type='ner'):
        """
           cadec: CADEC (Parser only. You must place the files)
           Extract CADEC dataset using torchtext.
        """
        logger.info('---------- CADEC = %s ---------' % (tag_type))
        train_file = mapping_files[opt.lang]
        # Setup fields with batch dimension first
        inputs_word = data.Field(
            batch_first=True,
            fix_length=opt.maxlen,
            lower=opt.lower,
            preprocessing=data.Pipeline(
                lambda w: '0' if opt.convert_digits and w.isdigit() else w))

        inputs_char_nesting = data.Field(
            tokenize=list, batch_first=True, fix_length=opt.maxlen)
        inputs_char = data.NestedField(inputs_char_nesting)

        inputs_case = data.Field(
            batch_first=True,
            fix_length=opt.maxlen,
            preprocessing=data.Pipeline(
                lambda w: self.getCasing(w)))

        labels = data.Field(
            batch_first=True,
            unk_token=None,
            fix_length=opt.maxlen)  # pad_token=None,
        # preprocessing=data.Pipeline(lambda w: labels_map[w]))

        id = data.Field(batch_first=True, use_vocab=False)

        self.fields = ([(('inputs_word',
                          'inputs_char',
                          'inputs_case'),
                         (inputs_word,
                          inputs_char,
                          inputs_case))] + [('labels',
                                             labels) if label == tag_type else (None,
                                                                                None) for label in ['ner']] + [('id',
                                                                                                                id)])

        # Load the data
        datafile = NERDataset.splits(
            path='.',
            train=train_file,
            separator='\t',
            encoding='utf-8',
            fields=tuple(self.fields))[0]

        self.train, self.val, self.test = datafile.split(
            split_ratio=[5610, 1000, 1000])
        return inputs_word, inputs_char, inputs_case, labels

    def check_ids(self, examples):  # no duplicate ids!
        a = [i.id[0] for i in examples]
        assert len(a) == len(set(a))

    def keep_duplicates(self):
        self.temp_train = data.Dataset(
            examples=self.train.examples,
            fields=self.fields)
        self.temp_val = data.Dataset(
            examples=self.val.examples,
            fields=self.fields)
        self.temp_test = data.Dataset(
            examples=self.test.examples,
            fields=self.fields)

    def reset(self):
        self.train = self.temp_train
        self.val = self.temp_val
        self.test = self.temp_test
        self.keep_duplicates()
        self.count_new = 0
        if(self.labeled != -1):
            # Create unlabeled dataset
            ratio = self.labeled / len(self.train)
            if ratio != 0:
                self.train, self.train_unlabeled = self.train.split(ratio)
            else:
                self.train_unlabeled = data.Dataset(
                    examples=self.train.examples, fields=self.fields)
                self.train.examples = []

    def batch_iter(self, batch_size):
        if(self.opt.adaptive_batch_size):
            batch_size = int(math.ceil(len(self.train) /
                                       self.opt.adaptive_batch_size))
        # Get iterators
        unlabeled_iter, _, _ = data.BucketIterator.splits(
            (self.train_unlabeled, self.val, self.test), batch_size=batch_size*self.opt.n_ubatches, shuffle=True,
            sort_key=lambda x: data.interleave_keys(len(x.inputs_word), len(x.inputs_char)),
            device=torch.device("cuda:" + str(self.gpu) if self.gpu != -1 else "cpu"))
        train_iter, val_iter, test_iter = data.BucketIterator.splits(
            (self.train, self.val, self.test), batch_size=batch_size, shuffle=True,
            sort_key=lambda x: data.interleave_keys(len(x.inputs_word), len(x.inputs_char)),
            device=torch.device("cuda:" + str(self.gpu) if self.gpu != -1 else "cpu"))
        train_iter.repeat = False
        return train_iter, val_iter, test_iter, unlabeled_iter

    def label(self, example):
        self.train.examples.append(example)
        for i in self.train_unlabeled.examples:
            if(i.id == example.id):
                assert i.inputs_word == example.inputs_word
                assert i.labels == example.labels
                self.train_unlabeled.examples.remove(i)
        #self.train_unlabeled.examples = [i for i in self.train_unlabeled.examples if i.id!=example.id]
        # self.train_unlabeled.examples.remove(example)
        self.count_new += 1

    def pseudo_label(self, example, model):
        # Add example with new label
        temp_example = None
        for i in self.train_unlabeled.examples:
            if(i.id == example.id):
                temp_example = i
                self.train_unlabeled.examples.remove(i)
        assert temp_example.inputs_word == example.inputs_word
        assert temp_example.labels == example.labels
        prediction = self.get_prediction(example, self.fields, model)
        temp_example.labels = prediction
        self.train.examples.append(temp_example)
        self.count_new += 1

    def sample_unlabeled(self, k_num):
        # TODO: remove sampling for unlabeled: cluster?
        # Random sample k points from D_pool
        unlabeled_pool = self.train_unlabeled
        indices = np.arange(len(unlabeled_pool.examples))
        np.random.shuffle(indices)
        sampled_examples = [example for count, example in enumerate(
            unlabeled_pool.examples) if count in indices[:k_num]]
        unlabeled_entries = data.Dataset(sampled_examples, self.fields)
        return unlabeled_entries

    def sample_validation(self, k_num):
        validation_pool = self.val
        indices = np.arange(len(validation_pool.examples))
        np.random.shuffle(indices)
        sampled_examples = [example for count, example in enumerate(
            validation_pool.examples) if count in indices[:k_num]]
        self.val.examples = sampled_examples
        logger.info('Sampled Validation size: %d' % (len(self.val)))
        # np.random.shuffle(dataset.val.examples)
        #dataset.val.examples = dataset.val.examples[:opt.labeled]
        #logger.info('Sampled Validation size: %d' % (len(dataset.val)))

    # https://github.com/mxhofer/Named-Entity-Recognition-BidirectionalLSTM-CNN-CoNLL/
    # define casing s.t. NN can use case information to learn patterns

    def getCasing(self, word):
        casing = 'other'
        numDigits = 0
        for char in word:
            if char.isdigit():
                numDigits += 1
        digitFraction = numDigits / float(len(word))
        if word.isdigit():  # Is a digit
            casing = 'numeric'
        elif digitFraction > 0.5:
            casing = 'mainly_numeric'
        elif word.islower():  # All lower case
            casing = 'allLower'
        elif word.isupper():  # All upper case
            casing = 'allUpper'
        elif word[0].isupper():  # is a title, initial char upper, then all lower
            casing = 'initialUpper'
        elif numDigits > 0:
            casing = 'contains_digit'

        return caseLookup[casing]

    def get_prediction(self, example, fields, model):  # Predict label
        pseudo_example = data.Dataset(examples=[example], fields=fields)
        pseudo_example = data.BucketIterator(
            dataset=pseudo_example,
            batch_size=1,
            repeat=False,
            shuffle=False,
            device=torch.device(
                "cuda:" + str(self.gpu) if self.gpu != -1 else "cpu"))
        assert len(pseudo_example) == 1
        if isinstance(model, torch.nn.Module):
            pad = model.wordrepr.tag_vocab.stoi['<pad>']
            pseudo_example = list(pseudo_example)[0]
            _, _, prediction = model(pseudo_example)
            y = list(
                filter(
                    lambda x: x != pad,
                    pseudo_example.labels.data.tolist()[0]))
        else:
            x, y = model.iter_to_xy(pseudo_example)
            _, _, prediction = model(x, y)
            y = list(filter(lambda x: x != '<pad>', y[0]))

        assert len(prediction) == 1
        prediction = lib.utils.indices2words(
            prediction, model.wordrepr.tag_vocab)
        #y = lib.utils.indices2words([y], model.wordrepr.tag_vocab)
        #tokens = lib.utils.indices2words([pseudo_example.inputs_word.data.tolist()[0]], model.wordrepr.word_vocab)
        prediction = prediction[0]
        # shrink prediction to same len as labels
        prediction = prediction[:len(y)]
        return prediction
