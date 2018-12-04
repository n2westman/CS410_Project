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

VAT, 12/04

1 epoch, 1000 labels + unlabeled data

INFO:train:Train loss: 1233.971, accuracy: 0.800, f1: 0.286, prec: 0.334, rec: 0.266
INFO:train:Validation loss: 61.168, accuracy: 0.949, f1: 0.760, prec: 0.750, rec: 0.770
INFO:train:First accuracy: 0.949, f1: 0.760, prec: 0.750, rec: 0.770
INFO:train:Test finished.

VAT Model, 12/03

IP = 0, Alpha = 0.2

1 epoch, full dataset

INFO:train:Train loss: 39.910, accuracy: 0.936, f1: 0.704, prec: 0.714, rec: 0.699
INFO:train:Validation loss: 27.016, accuracy: 0.962, f1: 0.825, prec: 0.851, rec: 0.801
INFO:train:First accuracy: 0.962, f1: 0.825, prec: 0.851, rec: 0.801
INFO:train:Test finished.




VAT Model, 12/03

IP = 1, Alpha = 1

7 epochs, 1000 labels

INFO:train:Train loss: 6.592, accuracy: 0.989, f1: 0.952, prec: 0.953, rec: 0.953
INFO:train:Validation loss: 64.700, accuracy: 0.934, f1: 0.677, prec: 0.696, rec: 0.660
INFO:train:First accuracy: 0.934, f1: 0.677, prec: 0.696, rec: 0.660

VAT Model, 12/03

IP = 1, Alpha = 0.2

7 epochs, 1000 labels

INFO:train:Train loss: 6.516, accuracy: 0.989, f1: 0.951, prec: 0.953, rec: 0.950
INFO:train:Validation loss: 62.328, accuracy: 0.937, f1: 0.692, prec: 0.708, rec: 0.677
INFO:train:First accuracy: 0.937, f1: 0.692, prec: 0.708, rec: 0.677
INFO:train:Test finished.
INFO:train:--- 581.359920979 seconds ---

VAT Model, 12/03

IP = 1, Alpha = 0.4

5 epochs, 1000 labels

INFO:train:Train loss: 10.561, accuracy: 0.983, f1: 0.927, prec: 0.932, rec: 0.923
INFO:train:Validation loss: 55.139, accuracy: 0.936, f1: 0.686, prec: 0.710, rec: 0.663
INFO:train:First accuracy: 0.936, f1: 0.686, prec: 0.710, rec: 0.663
