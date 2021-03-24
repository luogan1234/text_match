import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BiLSTMEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embedding = nn.Embedding(config.vocab_num, config.word_embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(config.word_embedding_dim, config.hidden_dim//2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(config.dropout_rate)
    
    def forward(self, batch):
        x = batch['inputs']
        x = self.word_embedding(x)
        h, _ = self.lstm(x)
        h = self.dropout(h)
        return h