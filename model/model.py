import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.bert_encoder import BERTEncoder

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = BERTEncoder(config)
        self.cls_fc = nn.Linear(config.hidden_dim, 1)
        self.mask_fc = nn.Linear(config.hidden_dim, config.vocab_num)
    
    def forward(self, batch):
        h = self.encoder(batch)  # [batch_size, seq_len, hiden_dim]
        self.cls_h = h[:, 0, :]
        cls_outputs = self.cls_fc(self.cls_h).squeeze(-1)
        mask_outputs = self.mask_fc(h)
        return cls_outputs, mask_outputs