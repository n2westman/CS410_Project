# CS410_Project
Semi-Supervised NLP

## Overview

This project will look at using new semi-supervised learning techniques to perform tasks such as Named Entity Recognition (NER) on datasets. We use the CONLL 2003 dataset here.

## Setup

You'll find data/ in the data subdirectory, and all source in the source/ subdirectory.

To install required python package, use pip:

```
pip install -r requirements.txt
```

Additionally, to run the auto-formatter, use autopep8:

```
autopep8 --in-place --aggressive --aggressive <filename>
```

## Running

To run, use train.py form the root directory. Logs should be printed to train.log.

```
python train.py
```

Arguments are listed in config.py, but some important ones include:

*  `-st_method=VAT` which will use the VAT implementation
*  `-st_method=MT` which will use the Mean Teachers implementation
*  `-st_method=BASELINE` which will use the baseline implementation
*  `-labeled=number` which set the number of labeled examples to use. Setting this value to -1 will use all labels available
*  `-batch_size=number` will set the batch size for training. The higher the number, the faster the training (assuming you have enough memory)
*  `-n_ubatches=number` the number of unlabeled batches to use at each step for the semi-supervised learning techniques. This will only create a consistency_loss, not a cross-entropy loss. The default is 3, which means 3xbatch_size examples will be used at each step.
*  `-end_epoch=number` will set the end epoch for model training.

Hyperparameters for VAT:

*  `-alpha=1.` will change alpha, the mutliplier of the consistency_loss, to 2. The default value is 0.2

## Known Issues

Model saving/loading hasn't yet been implemented.

## Results

Baseline Model, 11/28

1 epoch, full dataset

INFO:train:Train loss: 39.993, accuracy: 0.936, f1: 0.704, prec: 0.715, rec: 0.697
INFO:train:Validation loss: 26.658, accuracy: 0.961, f1: 0.816, prec: 0.839, rec: 0.794
INFO:train:First accuracy: 0.961, f1: 0.816, prec: 0.839, rec: 0.794


Baseline, 12/04

7 epoch, 1000 label

INFO:train:Train loss: 25.708, accuracy: 0.955, f1: 0.800, prec: 0.809, rec: 0.793
INFO:train:Validation loss: 54.083, accuracy: 0.919, f1: 0.606, prec: 0.600, rec: 0.611
INFO:train:First accuracy: 0.919, f1: 0.606, prec: 0.600, rec: 0.611

Baseline 12/06

5 epoch, all labels

INFO:train:Train loss: 11.762, accuracy: 0.986, f1: 0.941, prec: 0.945, rec: 0.938
INFO:train:Validation loss: 25.983, accuracy: 0.968, f1: 0.852, prec: 0.853, rec: 0.850
INFO:train:First accuracy: 0.968, f1: 0.852, prec: 0.853, rec: 0.850

VAT, 12/06

7 epoch, 1000 label, no unlabeled data
IP = 1, Alpha = 0.2

INFO:train:Train loss: 6.956, accuracy: 0.989, f1: 0.954, prec: 0.956, rec: 0.951
INFO:train:Validation loss: 62.344, accuracy: 0.938, f1: 0.700, prec: 0.738, rec: 0.666
INFO:train:First accuracy: 0.938, f1: 0.700, prec: 0.738, rec: 0.666

VAT, 12/06

7 epoch, 1000 label, no unlabeled data
IP = 1, Alpha = 1.0

INFO:train:Train loss: 9.580, accuracy: 0.991, f1: 0.959, prec: 0.961, rec: 0.957
INFO:train:Validation loss: 58.895, accuracy: 0.937, f1: 0.692, prec: 0.710, rec: 0.674
INFO:train:First accuracy: 0.937, f1: 0.692, prec: 0.710, rec: 0.674

VAT, 12/06

1 epoch, All labels
IP = 1, Alpha = 0.2

INFO:train:Train loss: 40.226, accuracy: 0.937, f1: 0.711, prec: 0.721, rec: 0.706
INFO:train:Validation loss: 26.973, accuracy: 0.960, f1: 0.813, prec: 0.836, rec: 0.791
INFO:train:First accuracy: 0.960, f1: 0.813, prec: 0.836, rec: 0.791

VAT 12/06

5 epoch, all labels
IP = 1, Alpha = 1.0

INFO:train:Train loss: 10.425, accuracy: 0.986, f1: 0.939, prec: 0.943, rec: 0.936
INFO:train:Validation loss: 28.262, accuracy: 0.968, f1: 0.854, prec: 0.865, rec: 0.842
INFO:train:First accuracy: 0.968, f1: 0.854, prec: 0.865, rec: 0.842

MT 12/12

7 Epoch, 1000 labels, no unlabeled

INFO:train:Train loss: 26.799, accuracy: 0.953, f1: 0.790, prec: 0.803, rec: 0.780
INFO:train:Validation loss: 51.292, accuracy: 0.925, f1: 0.630, prec: 0.649, rec: 0.611
INFO:train:First accuracy: 0.925, f1: 0.630, prec: 0.649, rec: 0.611
