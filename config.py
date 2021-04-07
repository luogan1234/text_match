import numpy as np
import torch
import math

class Config:
    def __init__(self, train, test, lr0, lr, bs, mask_w, seed, cpu):
        self.train = train
        self.test = test
        self.lr0 = lr0
        self.lr = lr
        self.bs = bs
        self.mask_w = mask_w
        self.seed = seed
        self.device = 'cpu' if cpu else 'cuda'
        
        self.path = '.' #'/data/luogan/text_match'
        self.dropout_rate = 0.1
        epoch_steps = 124999//self.bs+1
        self.warmup_steps = 2*epoch_steps
        self.training_steps = 30*epoch_steps
        self.warmup_steps0 = 5*epoch_steps
        self.training_steps0 = 75*epoch_steps
        self.hidden_dim = 768
    
    def learning_rate(self, stage):
        lr = self.lr0 if stage == 1 else self.lr
        return lr
    
    def batch_size(self, mode):
        bs = self.bs
        if not mode:
            bs *= 4
        return bs
    
    def store_name(self, seed=None):
        seed = self.seed if seed is None else seed
        return '{}_{}_{}_{}_{}_{}_{}'.format(self.train, self.test, self.lr0, self.lr, self.bs, self.mask_w, seed)
    
    def store_path(self, seed=None):
        return '{}/result/model_states/{}.pth'.format(self.path, self.store_name(seed))
    
    def feature_path(self):
        return '{}/result/features/{}_{}_{}.npy'.format(self.path, self.train, self.test, self.seed)
    
    def pretrain_path(self):
        return '{}/result/model_states/bert_embeddings_{}_{}.pth'.format(self.path, self.lr0, self.bs)
    
    def result_path(self):
        return 'result/result.txt'.format(self.path)
    
    def prediction_path(self):
        store_name = '{}_{}_{}_{}_{}'.format(self.train, self.test, self.lr0, self.lr, self.bs)
        return 'result/{}.csv'.format(store_name)
    
    def parameter_info(self):
        obj = {'train': self.train, 'learning_rate0': self.lr0, 'learning_rate': self.lr, 'batch_size': self.bs, 'warmup_steps0': self.warmup_steps0, 'warmup_steps': self.warmup_steps, 'training_steps0': self.training_steps0, 'training_steps': self.training_steps, 'mask_w': self.mask_w, 'seed': self.seed}
        return obj