import numpy as np
import os
import json
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import random
import copy
from torch.utils.data import Dataset, DataLoader, random_split

class MyDataset(Dataset):
    def __init__(self, file, is_train):
        self.data, self.labels, self.is_train = [], [], is_train
        self.vocabs, self.vocab_num = set(), 0
        with open('data/{}.tsv'.format(file), 'r') as f:
            for line in f:
                items = line.split('\t')
                seq1 = self.to_index(items[0])
                seq2 = self.to_index(items[1])
                self.data.append([seq1, seq2])
                self.vocabs = self.vocabs.union(seq1+seq2)
                self.vocab_num = max([self.vocab_num]+seq1+seq2)
                if is_train:
                    self.labels.append(int(items[2]))
                else:
                    self.labels.append(-1)
        self.vocab_num += 1
        print('Vocab number:', self.vocab_num)
    
    def to_index(self, seq):
        # [PAD], [unused1]...[unused99], [UNK], [CLS], [SEP], [MASK]
        # 0, 1..99, 100, 101, 102, 103
        seq = [int(v)+104 for v in seq.split(' ')]
        return seq
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = {'seq1': self.data[idx][0], 'seq2': self.data[idx][1], 'label': self.labels[idx]}
        return item

class MyCollation:
    def __init__(self, config, is_train):
        self.config = config
        self.is_train = is_train
    
    def mask(self, seq):
        res, label_res = [], []
        for token in seq:
            p = random.random()
            if self.is_train and p < 0.15:
                p /= 0.15
                label_res.append(token)
                if p < 0.8:
                    res.append(103)  # [MASK]
                elif p < 0.9:
                    res.append(random.sample(self.config.vocabs, 1)[0])
                else:
                    res.append(token)
            else:
                res.append(token)
                label_res.append(0)
        return res, label_res
    
    def __call__(self, data):
        inputs, segs, mask_labels, cls_labels = [], [], [], []
        max_len = 0
        for datum in data:
            max_len = max(max_len, len(datum['seq1'])+len(datum['seq2'])+3)
        for datum in data:
            seq1, label_seq1 = self.mask(datum['seq1'])
            seq2, label_seq2 = self.mask(datum['seq2'])
            if not self.is_train or random.randint(0, 1):
                input = [101]+seq1+[102]+seq2+[102]
                seg = [0]*(len(seq1)+2)+[1]*(len(seq2)+1)
                mask_label = [0]+label_seq1+[0]+label_seq2+[0]
            else:
                input = [101]+seq2+[102]+seq1+[102]
                seg = [0]*(len(seq2)+2)+[1]*(len(seq1)+1)
                mask_label = [0]+label_seq2+[0]+label_seq1+[0]
            input += [0]*(max_len-len(input))
            seg += [1]*(max_len-len(seg))
            mask_label += [0]*(max_len-len(mask_label))
            inputs.append(input)
            segs.append(seg)
            mask_labels.append(mask_label)
            cls_labels.append(datum['label'])
        inputs = torch.tensor(inputs, dtype=torch.long).to(self.config.device)
        segs = torch.tensor(segs, dtype=torch.long).to(self.config.device)
        res = {'inputs': inputs, 'segs': segs, 'mask_labels': mask_labels, 'cls_labels': cls_labels}
        return res

class InfiniteDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = super().__iter__()
            batch = next(self.iterator)
        return batch

class MyDataLoader:
    def __init__(self, config):
        self.train = MyDataset(config.train, True)
        self.test = MyDataset(config.test, False)
        config.vocabs = self.train.vocabs.union(self.test.vocabs)
        config.vocab_num = max(self.train.vocab_num, self.test.vocab_num, 25000)
        print('Unknown token number in test:', len(self.test.vocabs-self.train.vocabs))
        self.config = config
        self.fn_train = MyCollation(config, True)
        self.fn_eval = MyCollation(config, False)
    
    def get_train(self):
        n = len(self.train)
        d1, d2 = int(n*0.9), n-int(n*0.9)
        train, valid = random_split(self.train, [d1, d2])
        [test] = random_split(self.test, [len(self.test)])
        valid_unlabel = []
        for datum in valid:
            datum_new = copy.deepcopy(datum)
            datum_new['label'] = -1
            valid_unlabel.append(datum_new)
        train += test+valid_unlabel
        train = InfiniteDataLoader(train, self.config.batch_size(True), shuffle=True, collate_fn=self.fn_train)
        valid = DataLoader(valid, self.config.batch_size(False), shuffle=False, collate_fn=self.fn_eval)
        return train, valid
    
    def get_all(self):
        data = DataLoader(self.train+self.test, self.config.batch_size(False), shuffle=False, collate_fn=self.fn_eval)
        return data
    
    def get_predict(self):
        data = DataLoader(self.test, self.config.batch_size(False), shuffle=False, collate_fn=self.fn_eval)
        return data