import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers.optimization import get_linear_schedule_with_warmup
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
        self.scheduler.step()
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
    
    def init(self):
        self.model = Model(self.config)
        no_decay = ['bias', 'LayerNorm.weight']
        self.optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        print('model parameters number: {}.'.format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))
        self.model.to(self.config.device)
    
    def train(self):
        print('Train starts:')
        if os.path.exists(self.config.store_path()):
            print('Train done.')
            return
        with open(self.config.store_path(), 'w') as f:
            f.write('!')
        self.init()
        train, valid = self.data_loader.get_train()
        train_iter = iter(train)
        print('Train batch size {}, eval batch size {}.'.format(self.config.batch_size(True), self.config.batch_size(False)))
        print('Batch number: train {}, valid {}.'.format(len(train), len(valid)))
        if self.config.text_encoder == 'bert':
            if not os.path.exists(self.config.pretrain_path()):
                print('Stage 1:')
                self.optimizer = optim.AdamW(self.optimizer_grouped_parameters, lr=self.config.learning_rate(1))
                self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=self.config.warmup_steps0, num_training_steps=self.config.training_steps0)
                print('warmup steps: {}, training steps: {}'.format(self.config.warmup_steps0, self.config.training_steps0))
                for i, p in enumerate(self.model.encoder.bert.parameters()):
                    if i > 0:
                        p.requires_grad = False
                min_train_loss, epoch, global_steps = 1e16, 0, 0
                try:
                    while global_steps < self.config.training_steps0:
                        epoch += 1
                        train_mask_loss = 0.0
                        train_tqdm = tqdm.tqdm(range(len(train)))
                        train_tqdm.set_description('Epoch {} | train_mask_loss: {:.4f}'.format(epoch, 0))
                        for steps in train_tqdm:
                            batch = next(train_iter)
                            loss, cls_loss, mask_loss = self.train_one_step(batch, True)
                            train_mask_loss += mask_loss
                            train_tqdm.set_description('Epoch {} | train_mask_loss: {:.4f}'.format(epoch, mask_loss))
                        steps += 1
                        global_steps += steps
                        train_mask_loss /= steps
                        print('Average train_mask_loss: {:.4f}.'.format(train_mask_loss))
                        if train_mask_loss < min_train_loss:
                            min_train_loss = train_mask_loss
                            word_embeddings = copy.deepcopy(self.model.encoder.bert.embeddings.word_embeddings.state_dict())
                            mask_fc = copy.deepcopy(self.model.mask_fc.state_dict())
                except KeyboardInterrupt:
                    train_tqdm.close()
                    print('Exiting from training early.')
                    os.remove(self.config.store_path())
                    return
                with open(self.config.pretrain_path(), 'wb') as f:
                    torch.save([word_embeddings, mask_fc], f)
                for i, p in enumerate(self.model.encoder.bert.parameters()):
                    if i > 0:
                        p.requires_grad = True
            print('Stage 2:')
            self.init()
            with open(self.config.pretrain_path(), 'rb') as f:
                [word_embeddings, mask_fc] = torch.load(f)
            self.model.encoder.bert.embeddings.word_embeddings.load_state_dict(word_embeddings)
            #self.model.mask_fc.load_state_dict(mask_fc)
        max_valid_auc, epoch, global_steps = 0.0, 0, 0
        best_scores = {}
        self.optimizer = optim.AdamW(self.optimizer_grouped_parameters, lr=self.config.learning_rate(2))
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=self.config.warmup_steps, num_training_steps=self.config.training_steps)
        print('warmup steps: {}, training steps: {}'.format(self.config.warmup_steps, self.config.training_steps))
        if self.config.finetune:
            print('Finetune from existing paramters.')
            file = self.config.store_path(False)
            if os.path.exists(file):
                with open(file, 'rb') as f:
                    best_para = torch.load(f)
                self.model.load_state_dict(best_para)
        try:
            while global_steps < self.config.training_steps:
                epoch += 1
                train_loss, train_cls_loss, train_mask_loss = 0.0, 0.0, 0.0
                train_tqdm = tqdm.tqdm(range(len(train)))
                train_tqdm.set_description('Epoch {} | train_loss: {:.4f}'.format(epoch, 0))
                for steps in train_tqdm:
                    batch = next(train_iter)
                    loss, cls_loss, mask_loss = self.train_one_step(batch, False)
                    train_loss += loss
                    train_cls_loss += cls_loss
                    train_mask_loss += mask_loss
                    train_tqdm.set_description('Epoch {} | train_loss: {:.4f}'.format(epoch, loss))
                steps += 1
                global_steps += steps
                print('Average train_loss: {:.4f}, train_cls_loss: {:.4f}, train_mask_loss: {:.4f}.'.format(train_loss/steps, train_cls_loss/steps, train_mask_loss/steps))
                valid_loss, scores = self.evaluate(valid, 'valid')
                if scores['auc'] > max_valid_auc:
                    max_valid_auc = scores['auc']
                    best_scores = copy.deepcopy(scores)
                    best_para = copy.deepcopy(self.model.state_dict())
        except KeyboardInterrupt:
            train_tqdm.close()
            print('Exiting from training early.')
            os.remove(self.config.store_path())
            return
        print('Train finished, max valid auc {:.4f}, stop at epoch {}.'.format(max_valid_auc, epoch))
        with open(self.config.store_path(), 'wb') as f:
            torch.save(best_para, f)
        result_path = self.config.result_path()
        with open(result_path, 'a', encoding='utf-8') as f:
            obj = self.config.parameter_info()
            obj.update(best_scores)
            f.write(json.dumps(obj)+'\n')
    
    def predict(self):
        print('Predict starts:')
        self.model = Model(self.config)
        self.model.to(self.config.device)
        print('model parameters number: {}.'.format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))
        predicts = []
        for seed in range(100):
            file = self.config.store_path(seed=seed)
            if not os.path.exists(file):
                continue
            print('Ensemble id:', seed)
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
        with open(self.config.prediction_path(), 'w') as f:
            f.write('\n'.join([str(v) for v in predicts]))