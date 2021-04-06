import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertModel, BertConfig

class BERTEncoder(nn.Module):        
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.bert.resize_token_embeddings(104)
        self.bert.resize_token_embeddings(config.vocab_num)
    
    def forward(self, batch):
        x, y = batch['inputs'], batch['segs']
        outputs = self.bert(x, attention_mask=(x>0), token_type_ids=y)
        h = outputs.last_hidden_state
        return h