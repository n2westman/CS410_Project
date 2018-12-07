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


## Results

Baseline Model, 11/28

1 epoch, full dataset

INFO:train:Train loss: 39.993, accuracy: 0.936, f1: 0.704, prec: 0.715, rec: 0.697
INFO:train:Validation loss: 26.658, accuracy: 0.961, f1: 0.816, prec: 0.839, rec: 0.794
INFO:train:First accuracy: 0.961, f1: 0.816, prec: 0.839, rec: 0.794


Baseline, 12/04

7 epoch, 1000 label

INFO:train:Train loss: 7.434, accuracy: 0.988, f1: 0.949, prec: 0.951, rec: 0.947
INFO:train:Validation loss: 59.172, accuracy: 0.936, f1: 0.688, prec: 0.703, rec: 0.675
INFO:train:First accuracy: 0.936, f1: 0.688, prec: 0.703, rec: 0.675

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
