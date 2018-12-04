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

VAT Model, 12/03

IP = 0, Alpha = 0.2

1 epoch, full dataset

INFO:train:Train loss: 39.910, accuracy: 0.936, f1: 0.704, prec: 0.714, rec: 0.699
INFO:train:Validation loss: 27.016, accuracy: 0.962, f1: 0.825, prec: 0.851, rec: 0.801
INFO:train:First accuracy: 0.962, f1: 0.825, prec: 0.851, rec: 0.801
INFO:train:Test finished.

VAT Model, 12/03

IP = 0, Alpha = 0.2

7 epochs, 1000 labels

INFO:train:Train loss: 5.922, accuracy: 0.991, f1: 0.958, prec: 0.958, rec: 0.958
INFO:train:Validation loss: 68.470, accuracy: 0.934, f1: 0.674, prec: 0.707, rec: 0.643
INFO:train:First accuracy: 0.934, f1: 0.674, prec: 0.707, rec: 0.643
INFO:train:Test finished.
INFO:train:--- 581.359920979 seconds ---
