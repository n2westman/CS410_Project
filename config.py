#__date__ = 6/14/18
#__time__ = 4:08 PM
#__author__ = isminilourentzou

import argparse
import random
import torch
import os
import numpy as np
from torch import cuda

parser = argparse.ArgumentParser()

## Data
parser.add_argument('-train', type=str, default="en", help="training phase")
parser.add_argument('-test', type=str, default="en;es;de;nl", help="testing phase")
parser.add_argument("-maxlen", type=int, default=100)
parser.add_argument('-maxvocab', type=int, default=20000, help='Max number of words in vocab')
parser.add_argument('-maxcharvocab', type=int, default=85, help='Max number of chars in vocab')
parser.add_argument('-lower', action='store_true', default=False ,help='Lowercase sequences')
parser.add_argument("-convert_digits", action='store_true', default=True, help='Map all numbers to 0')

## Model
parser.add_argument('-plainCRF', action='store_true', default=False,help='Use plain CRF, if so, all other model arguments not used')
parser.add_argument('-brnn', action='store_true', default=True,help='Make RNNs bidirectional')
parser.add_argument("-dropout", type=float, default=0.2)
parser.add_argument("-hidden_dim", type=int, default=100)
parser.add_argument("-nlayers", type=int, default=1)
parser.add_argument("-word_emb_dim", type=int, default=100, help='Word embedding dimension, not used if loading pretrained')
parser.add_argument('-pre_embs', action='store_true', default=False, help='Use multilingual pretrained word embs')
parser.add_argument("-char_emb_dim", type=int, default=25, help='Char embedding dimension, chars not used if None')
parser.add_argument("-char_hidden_dim", type=int, default=50, help='Char layer hidden dimension')
parser.add_argument("-case_emb_dim", type=int, default=20, help='Casing embedding dimension, casing not used if None')

## Training
parser.add_argument('-batch_size', type=int, default=16, help='Training batch size')
parser.add_argument('-adaptive_batch_size', type=int, default=None, help='Adaptive batch size, #instances/#batches where #batches is defined by user')
parser.add_argument('-start_epoch', type=int, default=1, help='Epoch to start training.')
parser.add_argument('-end_epoch', type=int, default=1, help='Number of supervised learning epochs')
parser.add_argument('-optim', type=str, default='adam', choices=['sgd', 'adam', 'adagrad', 'adadelta'], help='Optimization method.')
parser.add_argument('-lr', type=float, default=0.01, help='Initial learning rate')
parser.add_argument('-max_grad_norm', type=float, default=5, help='Clip gradients by max global gradient norm. See https://arxiv.org/abs/1211.5063')
parser.add_argument('-learning_rate_decay', type=float, default=0.05, help='Multiply learning with this value after -start_decay_after epochs')
parser.add_argument('-start_decay_after', type=int, default=None, help='Decay learning rate AFTER this epoch')

## GPU and others
parser.add_argument('-seed', type=int, default=0, help='Random seed')
parser.add_argument('-gpu', type=int, default=-1)
parser.add_argument('-log_interval', type=int, default=1, help='Print stats after that many training steps')
parser.add_argument("-test_every", type=int, default=None, help='Test every k training episodes')

## Args for saving and loading model
parser.add_argument('-logfolder', type=str, default=None, help='Log to folder')
parser.add_argument('-load_from', type=str, help='Path to a checkpoint')
parser.add_argument('-save_dir', type=str, default='exps', help='Path to a checkpoint')

## Active Learning
parser.add_argument('-al_method', type=str, default='us', help='[us|rs|bald|nb|ds]')
parser.add_argument('-st_method', type=str, default='VAT', help='VAT')
parser.add_argument("-alpha", type=float, default=0.2, help="Used in VAT")
parser.add_argument('-nsamples', type=int, default=100, help='number of samples in batch for heuristics AL')
parser.add_argument('-labeled', type=int, default=100, help='Number of initial labeled examples')
parser.add_argument('-k_num', type=int, default=100, help="number of unlabeled examples to initially sample")
parser.add_argument('-budget', type=int, default=100, help="require a budget for annotating")
parser.add_argument('-episode', type=int, default=2, help="require a maximum number of playing the game")
parser.add_argument('-policy_hidden', type=int, default=100, help='Hidden layer size for policy network')
parser.add_argument('-epsilon', type=float, default=0.2, help='epsilon-greedy')
parser.add_argument('-buffer', type=int, default=10000, help='number of previous transitions to remember')
parser.add_argument('-policy_nepochs', type=int, default=1, help='Number of epochs for policy training')
parser.add_argument('-policy_batch_size', type=int, default=32, help='batch size for policy + train after buffer has that many transitions saved')
parser.add_argument('-constraint', type=str, default=None, help='[orth|l1|both] additional loss penalty - increased diversity: orthogonality, l1 or both')
parser.add_argument('-penalize_same', type=str, default=None, help='[penalty|stop], if same action for both is chosen = penalty: -1 reward, stop:-1 + stop episode')

opt = parser.parse_args()

def additional_args(opt):
    # Set seed
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    np.random.seed(opt.seed)
    # Set cuda
    opt.cuda = (opt.gpu != -1)
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with -gpu 1")
    if opt.cuda: cuda.set_device(opt.gpu)
    if(opt.logfolder):
        opt.save_dir = os.path.join(opt.save_dir, opt.logfolder)
    if opt.save_dir and not os.path.exists(opt.save_dir): os.makedirs(opt.save_dir)
