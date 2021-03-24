import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.bilstm_encoder import BiLSTMEncoder
from model.bert_encoder import BERTEncoder

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.text_encoder == 'bilstm':
            encoder = BiLSTMEncoder(config)
        if config.text_encoder in ['bert']:
            encoder = BERTEncoder(config)
        self.encoder = encoder
        self.cls_fc = nn.Linear(config.hidden_dim, 1)
        self.mask_fc = nn.Linear(config.hidden_dim, config.vocab_num)
    
    def forward(self, batch):
        h = self.encoder(batch)  # [batch_size, seq_len, hiden_dim]
        cls_outputs = self.cls_fc(h[:, 0, :]).squeeze(-1)
        mask_outputs = self.mask_fc(h)
        return cls_outputs, mask_outputs