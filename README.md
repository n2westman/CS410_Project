# CS410_Project
Semi-Supervised NLP

## Overview

This project will look at using new semi-supervised learning techniques to perform tasks such as Named Entity Recognition (NER) on datasets. We use the CONLL 2003 dataset here.

Before going further, a huge thank you to Ismini Lourentzou for helping throughout this project!

## Setup

You'll find all source in the lib/ subdirectory, with pip requirements in requirements.txt. This code was written using Python 2.7.14.

To install required python package, use pip:

```
pip install -r requirements.txt
```

Additionally, you'll need to provide the data. Thanks to isminilourentzou@ for helping me obtain the dataset. In particular, all this code was tested on CONLL 2003 english data. The code expects to find examples at `./data/conll2003/`.

## Running

To run, use train.py form the root directory. Logs should be printed to train.log.

```
python train.py & tail -f train.log
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

## Test Results

Baseline Model, 11/28

1 epoch, full dataset

Train loss: 39.993, accuracy: 0.936, f1: 0.704, prec: 0.715, rec: 0.697
Validation loss: 26.658, accuracy: 0.961, f1: 0.816, prec: 0.839, rec: 0.794
First accuracy: 0.961, f1: 0.816, prec: 0.839, rec: 0.794


Baseline, 12/04

7 epoch, 1000 label

Train loss: 25.708, accuracy: 0.955, f1: 0.800, prec: 0.809, rec: 0.793
Validation loss: 54.083, accuracy: 0.919, f1: 0.606, prec: 0.600, rec: 0.611
First accuracy: 0.919, f1: 0.606, prec: 0.600, rec: 0.611

Baseline 12/06

5 epoch, all labels

Train loss: 11.762, accuracy: 0.986, f1: 0.941, prec: 0.945, rec: 0.938
Validation loss: 25.983, accuracy: 0.968, f1: 0.852, prec: 0.853, rec: 0.850
First accuracy: 0.968, f1: 0.852, prec: 0.853, rec: 0.850

VAT, 12/06

7 epoch, 1000 label, no unlabeled data
IP = 1, Alpha = 0.2

Train loss: 6.956, accuracy: 0.989, f1: 0.954, prec: 0.956, rec: 0.951
Validation loss: 62.344, accuracy: 0.938, f1: 0.700, prec: 0.738, rec: 0.666
First accuracy: 0.938, f1: 0.700, prec: 0.738, rec: 0.666

VAT, 12/06

7 epoch, 1000 label, no unlabeled data
IP = 1, Alpha = 1.0

Train loss: 9.580, accuracy: 0.991, f1: 0.959, prec: 0.961, rec: 0.957
Validation loss: 58.895, accuracy: 0.937, f1: 0.692, prec: 0.710, rec: 0.674
First accuracy: 0.937, f1: 0.692, prec: 0.710, rec: 0.674

VAT, 12/06

1 epoch, All labels
IP = 1, Alpha = 0.2

Train loss: 40.226, accuracy: 0.937, f1: 0.711, prec: 0.721, rec: 0.706
Validation loss: 26.973, accuracy: 0.960, f1: 0.813, prec: 0.836, rec: 0.791
First accuracy: 0.960, f1: 0.813, prec: 0.836, rec: 0.791

VAT 12/06

5 epoch, all labels
IP = 1, Alpha = 1.0

Train loss: 10.425, accuracy: 0.986, f1: 0.939, prec: 0.943, rec: 0.936
Validation loss: 28.262, accuracy: 0.968, f1: 0.854, prec: 0.865, rec: 0.842
First accuracy: 0.968, f1: 0.854, prec: 0.865, rec: 0.842

MT 12/12

7 Epoch, 1000 labels, no unlabeled

Train loss: 26.799, accuracy: 0.953, f1: 0.790, prec: 0.803, rec: 0.780
Validation loss: 51.292, accuracy: 0.925, f1: 0.630, prec: 0.649, rec: 0.611
First accuracy: 0.925, f1: 0.630, prec: 0.649, rec: 0.611


## Final Results

### All Labels

For all labels, we trained on 5 epochs. The VAT model performed the best, and I hypothesize that the Mean Teachers implementation, which did not perform as well, needed more time to train.

#### Baseline (0.852)
Train loss: 11.762, accuracy: 0.986, f1: 0.941, prec: 0.945, rec: 0.938
Validation loss: 25.983, accuracy: 0.968, f1: 0.852, prec: 0.853, rec: 0.850
First accuracy: 0.968, f1: 0.852, prec: 0.853, rec: 0.850

#### VAT *(0.866)*
Train loss: 14.839, accuracy: 0.988, f1: 0.951, prec: 0.957, rec: 0.946
Validation loss: 21.556, accuracy: 0.972, f1: 0.866, prec: 0.879, rec: 0.854
First accuracy: 0.972, f1: 0.866, prec: 0.879, rec: 0.854

#### MT (0.713)
Train loss: 37.919, accuracy: 0.933, f1: 0.678, prec: 0.690, rec: 0.669
Validation loss: 35.430, accuracy: 0.942, f1: 0.713, prec: 0.724, rec: 0.702
First accuracy: 0.942, f1: 0.713, prec: 0.724, rec: 0.702

### 1000 Labels + Unlabeled Data

Unlabeled data is added at 3x batch size per iteration, to help with the training process. The baseline model can't use that extra data, but the two semi-supervised models can!

#### Baseline (0.616)
Train loss: 25.708, accuracy: 0.955, f1: 0.800, prec: 0.809, rec: 0.793
Validation loss: 54.083, accuracy: 0.919, f1: 0.606, prec: 0.600, rec: 0.611
First accuracy: 0.919, f1: 0.606, prec: 0.600, rec: 0.611

#### VAT *(0.697)*
Train loss: 17.569, accuracy: 0.990, f1: 0.955, prec: 0.960, rec: 0.950
Validation loss: 44.116, accuracy: 0.938, f1: 0.697, prec: 0.727, rec: 0.670
First accuracy: 0.938, f1: 0.697, prec: 0.727, rec: 0.670

#### MT (0.621)
Train loss: 24.607, accuracy: 0.955, f1: 0.792, prec: 0.802, rec: 0.785
Validation loss: 55.070, accuracy: 0.924, f1: 0.621, prec: 0.648, rec: 0.597
First accuracy: 0.924, f1: 0.621, prec: 0.648, rec: 0.597

### 1000 Labels, No unlabeled

#### VAT (0.701)
Train loss: 18.612, accuracy: 0.990, f1: 0.957, prec: 0.964, rec: 0.952
Validation loss: 42.629, accuracy: 0.939, f1: 0.701, prec: 0.722, rec: 0.681
First accuracy: 0.939, f1: 0.701, prec: 0.722, rec: 0.681

#### MT (0.607)
Train loss: 25.383, accuracy: 0.953, f1: 0.794, prec: 0.801, rec: 0.789
Validation loss: 54.801, accuracy: 0.920, f1: 0.607, prec: 0.621, rec: 0.593
First accuracy: 0.920, f1: 0.607, prec: 0.621, rec: 0.593
