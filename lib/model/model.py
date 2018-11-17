#__date__ = 6/14/18
#__time__ = 4:08 PM
#__author__ = isminilourentzou


import torch.nn as nn
from .crf import CRF

class Model(nn.Module):
    def __init__(self, opt, wordrepr):
        super(Model, self).__init__()
        self.opt = opt
        self.wordrepr = wordrepr
        self.lstm = getattr(nn, 'GRU')(input_size=wordrepr.input_size, hidden_size=opt.hidden_dim // 2 if opt.brnn else opt.hidden_dim,
                                                batch_first=True, num_layers=opt.nlayers, bidirectional=opt.brnn)
        self.droplstm = nn.Dropout(opt.dropout)
        self.hidden2tag = nn.Linear(opt.hidden_dim, wordrepr.outsize)
        self.crf_tagger = CRF(wordrepr.outsize)

    def forward(self, batch):
        word_represent = self.wordrepr(batch)
        hidden = None
        lstm_out, hidden = self.lstm(word_represent, hidden)
        feature_out = self.droplstm(lstm_out.transpose(1, 0))
        outputs = self.hidden2tag(feature_out)
        assert outputs.size(2) == len(self.wordrepr.tag_vocab)
        #print('outputs', outputs.size())
        #print('batch.labels', batch.labels.size())
        llikelihood = - self.crf_tagger.forward(outputs, batch.labels.transpose(1, 0)) if batch.labels is not None else -1
        score, result = self.crf_tagger.decode(outputs)
        return llikelihood, score, result

    def get_intermediate_layer(self, layer_index, batch):
        word_represent = self.wordrepr(batch)
        hidden = None
        lstm_out, hidden = self.lstm(word_represent, hidden)
        feature_out = self.droplstm(lstm_out.transpose(1, 0))
        outputs = self.hidden2tag(feature_out)
        score, result = self.crf_tagger.decode(outputs)
        intermediate_layers = [word_represent, feature_out, outputs, result]
        if(layer_index>len(intermediate_layers)):
            raise KeyError('Index {} biggen than number of layers:{}'.format(layer_index, len(intermediate_layers)))
        return intermediate_layers[layer_index]
