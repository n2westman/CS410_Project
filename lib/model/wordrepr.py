#__date__ = 6/14/18
#__time__ = 4:08 PM
#__author__ = isminilourentzou


import torch
import torch.nn as nn
import torch.nn.functional as F

class WordRepr(nn.Module):
    def __init__(self, opt, vocabs):
        super(WordRepr, self).__init__()
        self.opt = opt
        self.word_vocab, self.char_vocab, self.case_vocab, self.tag_vocab = vocabs
        self.outsize = len(self.tag_vocab.itos)
        self.input_size = 0
        self.word_drop = nn.Dropout(opt.dropout)
        if(self.word_vocab.vectors is not None):
            self.word_embedding = nn.Embedding(len(self.word_vocab), self.word_vocab.vectors.shape[1])
            self.word_embedding.weight.data.copy_(self.word_vocab.vectors)
            self.input_size += self.word_vocab.vectors.shape[1]
        else:
            self.word_embedding = nn.Embedding(len(self.word_vocab),self.opt.word_emb_dim)
            self.input_size += self.opt.word_emb_dim

        if(self.opt.char_emb_dim):
            self.char_drop = nn.Dropout(opt.dropout)
            self.char_embedding = nn.Embedding(len(self.char_vocab), self.opt.char_emb_dim)
            self.char_cnn = nn.Conv1d(self.opt.char_emb_dim, opt.char_hidden_dim, kernel_size=3, padding=1)
            self.input_size += self.opt.char_hidden_dim

        if(self.opt.case_emb_dim):
            self.case_drop = nn.Dropout(opt.dropout)
            self.case_embedding = nn.Embedding(len(self.case_vocab), self.opt.case_emb_dim)
            self.input_size += self.opt.case_emb_dim

    def forward(self, batch):
        word_represent = self.word_embedding(batch.inputs_word)
        word_represent =  self.word_drop(word_represent)

        if(self.opt.char_emb_dim):
            char_embeds = self.char_embedding(batch.inputs_char)
            char_embeds = char_embeds.transpose(2,1).contiguous()
            char_embeds = char_embeds.view(-1, char_embeds.size(-2), char_embeds.size(-1)).transpose(2,1) #(batch_size*seq_len) x char_emb_dim x char_len
            char_cnn_out = self.char_cnn(char_embeds)
            char_cnn_out = F.max_pool1d(char_cnn_out, char_cnn_out.size(2)).view(word_represent.size(0),word_represent.size(1),-1)
            char_cnn_out = self.char_drop(char_cnn_out)
            word_represent = torch.cat([word_represent, char_cnn_out], 2)

        if(self.opt.case_emb_dim):
            case_embeds = self.case_embedding(batch.inputs_case)
            case_embeds = self.case_drop(case_embeds)
            word_represent = torch.cat([word_represent, case_embeds], 2)

        return word_represent
