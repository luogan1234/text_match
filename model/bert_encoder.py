import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertModel, BertConfig

class BERTEncoder(nn.Module):        
    def __init__(self, config):
        super().__init__()
        #bert_config = BertConfig(vocab_size=config.vocab_num, hidden_size=config.hidden_dim, num_hidden_layers=config.num_hidden_layers, num_attention_heads=config.num_attention_heads, intermediate_size=config.intermediate_size)
        #self.bert = BertModel(bert_config)
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.bert.resize_token_embeddings(104)
        self.bert.resize_token_embeddings(config.vocab_num)
    
    def forward(self, batch):
        x, y = batch['inputs'], batch['segs']
        outputs = self.bert(x, attention_mask=(x>0), token_type_ids=y)
        h = outputs.last_hidden_state
        return h