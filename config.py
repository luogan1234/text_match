import numpy as np
import torch
import math

class Config:
    def __init__(self, train, test, text_encoder, ensemble_num, word_embedding_dim, hidden_dim, num_hidden_layers, num_attention_heads, intermediate_size, no_pretrain, seed, cpu):
        self.train = train
        self.test = test
        self.text_encoder = text_encoder
        self.ensemble_num = ensemble_num
        self.pretrain = not no_pretrain
        self.seed = seed
        self.device = 'cpu' if cpu else 'cuda'
        
        self.feature_dim = 128
        self.dropout_rate = 0.1
        self.max_steps = 5000
        self.word_embedding_dim = word_embedding_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
    
    def lr(self):
        if self.text_encoder == 'bert':
            lr = 2e-5
        else:
            lr = 1e-3
        return lr
    
    def batch_size(self, mode):
        if self.text_encoder == 'bert':
            batch_size = 32
        else:
            batch_size = 32
        if not mode:
            batch_size *= 4
        return batch_size
    
    def early_stop_time(self):
        early_stop_time = 8
        return early_stop_time
    
    def store_name(self, id=0):
        return '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(self.train, self.test, self.text_encoder, self.ensemble_num, self.word_embedding_dim, self.hidden_dim, self.num_hidden_layers, self.num_attention_heads, self.intermediate_size, self.pretrain, id, self.seed)
    
    def parameter_info(self, id=0):
        obj = {'train': self.train, 'test': self.test, 'text_encoder': self.text_encoder, 'ensemble_num': self.ensemble_num, 'word_embedding_dim': self.word_embedding_dim, 'hidden_dim': self.hidden_dim, 'num_hidden_layers': self.num_hidden_layers, 'num_attention_heads': self.num_attention_heads, 'intermediate_size': self.intermediate_size, 'pretrain': self.pretrain, 'id': id, 'seed': self.seed}
        return obj