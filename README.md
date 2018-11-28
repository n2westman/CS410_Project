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
