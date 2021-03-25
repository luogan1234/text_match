import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
import os
import json
import numpy as np
import tqdm
import random
import transformers
from model.model import Model
import time
import pickle
import sys
import copy

class Processor(object):
    def __init__(self, data_loader, config):
        self.data_loader = data_loader
        self.config = config
    
    def bce_loss(self, outputs, labels):
        labels = torch.tensor(labels, dtype=torch.float).to(self.config.device)
        loss = F.binary_cross_entropy_with_logits(outputs, labels, labels>=0)
        return loss
    
    def ce_loss(self, outputs, labels):
        labels = torch.tensor(labels, dtype=torch.long).to(self.config.device)
        loss = F.cross_entropy(outputs.transpose(1, 2), labels, ignore_index=0)
        return loss
    
    def train_one_step(self, batch, pretrain):
        cls_outputs, mask_outputs = self.model(batch)
        cls_loss = self.bce_loss(cls_outputs, batch['cls_labels'])
        mask_loss = self.ce_loss(mask_outputs, batch['mask_labels'])
        if pretrain:
            loss = mask_loss
        else:
            loss = cls_loss+mask_loss
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item(), cls_loss.item(), mask_loss.item()
    
    def eval_one_step(self, batch):
        with torch.no_grad():
            cls_outputs, mask_outputs = self.model(batch)
            loss = self.bce_loss(cls_outputs, batch['cls_labels']).item()
            outputs = torch.sigmoid(cls_outputs).detach().cpu().numpy()
        return outputs, loss
    
    def evaluate(self, data, flag):
        self.model.eval()
        trues, preds = [], []
        eval_loss = 0
        eval_tqdm = tqdm.tqdm(data, total=len(data))
        eval_tqdm.set_description('eval_loss: {:.4f}'.format(0))
        for batch in eval_tqdm:
            outputs, loss = self.eval_one_step(batch)
            for j in range(len(outputs)):
                true = batch['cls_labels'][j]
                pred = outputs[j]
                trues.append(true)
                preds.append(pred)
            eval_loss += loss
            eval_tqdm.set_description('eval_loss: {:.4f}'.format(loss))
        eval_loss /= len(data)
        self.model.train()
        if trues:
            pairs = list(zip(trues, preds))
            pairs.sort(key=lambda x: x[1])
            rank_sum, pos_num, neg_num = 0, 0, 0
            for i, pair in enumerate(pairs):
                if pair[0] == 1:
                    pos_num += 1
                    rank_sum += i
                else:
                    neg_num += 1
            auc = (rank_sum-pos_num*(pos_num+1)//2)/(pos_num*neg_num)
            trues, preds = np.array(trues), np.array(preds)>0.5
            f1 = f1_score(trues, preds, average='micro')
            print('Average {} loss: {:.4f}, auc: {:.4f}, f1: {:.4f}.'.format(flag, eval_loss, auc, f1))
        else:
            auc = f1 = None
        score = {'auc': auc, 'f1': f1}
        return eval_loss, score
    
    def train(self):
        print('Train starts:')
        for id in range(self.config.ensemble_num):
            print('Ensemble id:', id)
            self.model = Model(self.config)
            print('model parameters number: {}.'.format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))
            self.model.to(self.config.device)
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config.lr())
            best_para = self.model.state_dict()
            train, valid = self.data_loader.get_train()
            train_iter = iter(train)
            print('Train batch size {}, eval batch size {}.'.format(self.config.batch_size(True), self.config.batch_size(False)))
            print('Batch number: train {}, valid {}.'.format(len(train), len(valid)))
            if self.config.text_encoder == 'bert':
                if os.path.exists('data/bert_embeddings.pth'):
                    with open('data/bert_embeddings.pth', 'rb') as f:
                        best_para = torch.load(f)
                    self.model.encoder.bert.embeddings.word_embeddings.load_state_dict(best_para)
                else:
                    print('Stage 1:')
                    for i, p in enumerate(self.model.encoder.bert.parameters()):
                        if i > 0:
                            p.requires_grad = False
                    min_train_loss, patience, iteration = 1e16, 0, 0
                    try:
                        while patience <= self.config.early_stop_time() and iteration < 40:
                            iteration += 1
                            train_mask_loss = 0.0
                            train_tqdm = tqdm.tqdm(range(min(len(train), self.config.max_steps)))
                            train_tqdm.set_description('Iteration {} | train_mask_loss: {:.4f}'.format(iteration, 0))
                            for steps in train_tqdm:
                                batch = next(train_iter)
                                loss, cls_loss, mask_loss = self.train_one_step(batch, True)
                                train_mask_loss += mask_loss
                                train_tqdm.set_description('Iteration {} | train_mask_loss: {:.4f}'.format(iteration, mask_loss))
                            steps += 1
                            train_mask_loss /= steps
                            print('Average train_mask_loss: {:.4f}.'.format(train_mask_loss))
                            if train_mask_loss < min_train_loss:
                                patience = 0
                                min_train_loss = train_mask_loss
                                best_para = copy.deepcopy(self.model.encoder.bert.embeddings.word_embeddings.state_dict())
                            patience += 1
                    except KeyboardInterrupt:
                        train_tqdm.close()
                        print('Exiting from training early.')
                    with open('data/bert_embeddings.pth', 'wb') as f:
                        torch.save(best_para, f)
                    self.model.encoder.bert.embeddings.word_embeddings.load_state_dict(best_para)
                    for i, p in enumerate(self.model.encoder.bert.parameters()):
                        if i > 0:
                            p.requires_grad = True
                    print('Stage 2:')
            max_valid_auc, patience, iteration = 0.0, 0, 0
            try:
                while patience <= self.config.early_stop_time():
                    iteration += 1
                    train_loss, train_cls_loss, train_mask_loss = 0.0, 0.0, 0.0
                    train_tqdm = tqdm.tqdm(range(min(len(train), self.config.max_steps)))
                    train_tqdm.set_description('Iteration {} | train_loss: {:.4f}'.format(iteration, 0))
                    for steps in train_tqdm:
                        batch = next(train_iter)
                        loss, cls_loss, mask_loss = self.train_one_step(batch, False)
                        train_loss += loss
                        train_cls_loss += cls_loss
                        train_mask_loss += mask_loss
                        train_tqdm.set_description('Iteration {} | train_loss: {:.4f}'.format(iteration, loss))
                    steps += 1
                    print('Average train_loss: {:.4f}, train_cls_loss: {:.4f}, train_mask_loss: {:.4f}.'.format(train_loss/steps, train_cls_loss/steps, train_mask_loss/steps))
                    valid_loss, scores = self.evaluate(valid, 'valid')
                    if scores['auc'] > max_valid_auc:
                        patience = 0
                        max_valid_auc = scores['auc']
                        best_para = copy.deepcopy(self.model.state_dict())
                    patience += 1
            except KeyboardInterrupt:
                train_tqdm.close()
                print('Exiting from training early.')
            print('Train finished, max valid auc {:.4f}, stop at iteration {}.'.format(max_valid_auc, iteration))
            #self.model.load_state_dict(best_para)
            #test_loss, scores = self.evaluate(test, 'test')
            #print('Test finished, test loss {:.4f}.'.format(test_loss))
            with open('result/model_states/{}.pth'.format(self.config.store_name(id)), 'wb') as f:
                torch.save(best_para, f)
            result_path = 'result/result.txt'
            with open(result_path, 'a', encoding='utf-8') as f:
                obj = self.config.parameter_info(id)
                obj.update(scores)
                f.write(json.dumps(obj)+'\n')
    
    def predict(self):
        print('Predict starts:')
        self.model = Model(self.config)
        self.model.to(self.config.device)
        print('model parameters number: {}.'.format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))
        predicts = []
        for id in range(self.config.ensemble_num):
            file = 'result/model_states/{}.pth'.format(self.config.store_name(id))
            if not os.path.exists(file):
                continue
            print('Ensemble id:', id)
            with open(file, 'rb') as f:
                best_para = torch.load(f)
            self.model.load_state_dict(best_para)
            self.model.eval()
            data = self.data_loader.get_predict()
            predict_tqdm = tqdm.tqdm(data, total=len(data))
            predict = []
            for batch in predict_tqdm:
                outputs, loss = self.eval_one_step(batch)
                for j in range(len(outputs)):
                    predict.append(outputs[j])
            predicts.append(predict)
        predicts = np.mean(np.array(predicts), 0).tolist()
        with open('result/predictions/{}.csv'.format(self.config.store_name()), 'w') as f:
            f.write('\n'.join([str(v) for v in predicts]))