import numpy as np
import torch
import math

class Config:
    def __init__(self, train, test, text_encoder, word_embedding_dim, hidden_dim, num_hidden_layers, num_attention_heads, intermediate_size, lr0, lr, bs, no_pretrain, finetune, seed, cpu):
        self.train = train
        self.test = test
        self.text_encoder = text_encoder
        self.lr0 = lr0
        self.lr = lr
        self.bs = bs
        self.pretrain = not no_pretrain
        self.finetune = finetune
        self.seed = seed
        self.device = 'cpu' if cpu else 'cuda'
        
        self.path = '/data/luogan/text_match'
        self.feature_dim = 512
        self.dropout_rate = 0.1
        epoch_steps = 124999//self.bs+1
        self.warmup_steps = 2*epoch_steps
        self.training_steps = 30*epoch_steps
        self.warmup_steps0 = 5*epoch_steps
        self.training_steps0 = 75*epoch_steps
        self.word_embedding_dim = word_embedding_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
    
    def learning_rate(self, stage):
        lr = self.lr0 if stage == 1 else self.lr
        return lr
    
    def batch_size(self, mode):
        bs = self.bs
        if not mode:
            bs *= 4
        return bs
    
    def store_name(self, flag=True, seed=None):
        finetune = self.finetune and flag
        seed = self.seed if seed is None else seed
        return '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(self.train, self.test, self.text_encoder, self.word_embedding_dim, self.hidden_dim, self.num_hidden_layers, self.num_attention_heads, self.intermediate_size, self.lr0, self.lr, self.bs, self.pretrain, finetune, seed)
    
    def store_path(self, flag=True, seed=None):
        return '{}/result/model_states/{}.pth'.format(self.path, self.store_name(flag, seed))
    
    def pretrain_path(self):
        return '{}/result/model_states/bert_embeddings_{}_{}.pth'.format(self.path, self.lr0, self.bs)
    
    def result_path(self):
        return '{}/result/result.txt'.format(self.path)
    
    def prediction_path(self):
        store_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(self.train, self.test, self.text_encoder, self.word_embedding_dim, self.hidden_dim, self.num_hidden_layers, self.num_attention_heads, self.intermediate_size, self.lr0, self.lr, self.bs, self.pretrain, self.finetune)
        return '{}/result/predictions/{}.csv'.format(self.path, store_name)
    
    def parameter_info(self):
        obj = {'train': self.train, 'test': self.test, 'text_encoder': self.text_encoder, 'word_embedding_dim': self.word_embedding_dim, 'hidden_dim': self.hidden_dim, 'num_hidden_layers': self.num_hidden_layers, 'num_attention_heads': self.num_attention_heads, 'intermediate_size': self.intermediate_size, 'learning_rate0': self.lr0, 'learning_rate': self.lr, 'batch_size': self.bs, 'pretrain': self.pretrain, 'finetune': self.finetune, 'seed': self.seed}
        return obj