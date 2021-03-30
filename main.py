import argparse
from data_loader import MyDataLoader
from processor import Processor
from config import Config
import pickle
import os
import torch
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    
def main():
    if not os.path.exists('result/'):
        os.mkdir('result/')
    if not os.path.exists('result/model_states/'):
        os.mkdir('result/model_states/')
    if not os.path.exists('result/predictions/'):
        os.mkdir('result/predictions/')
    parser = argparse.ArgumentParser(description='Text_match')
    parser.add_argument('-train', type=str, default='train', choices=['train'])
    parser.add_argument('-test', type=str, default='testA', choices=['testA'])
    parser.add_argument('-text_encoder', type=str, default='bert', choices=['bert', 'bilstm'])
    parser.add_argument('-word_embedding_dim', type=int, default=128)
    parser.add_argument('-hidden_dim', type=int, default=768)
    parser.add_argument('-num_hidden_layers', type=int, default=12)
    parser.add_argument('-num_attention_heads', type=int, default=12)
    parser.add_argument('-intermediate_size', type=int, default=3072)
    parser.add_argument('-lr0', type=float, default=1e-4)
    parser.add_argument('-lr', type=float, default=3e-5)
    parser.add_argument('-bs', type=int, default=32)
    parser.add_argument('-no_pretrain', action='store_true', default=False)
    parser.add_argument('-finetune', action='store_true', default=False)
    parser.add_argument('-predict', action='store_true', default=False)
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-cpu', action='store_true')
    args = parser.parse_args()
    set_seed(args.seed)
    config = Config(args.train, args.test, args.text_encoder, args.word_embedding_dim, args.hidden_dim, args.num_hidden_layers, args.num_attention_heads, args.intermediate_size, args.lr0, args.lr, args.bs, args.no_pretrain, args.finetune, args.seed, args.cpu)
    data_loader = MyDataLoader(config)
    processor = Processor(data_loader, config)
    if not args.predict:
        processor.train()
    else:
        processor.predict()

if __name__ == '__main__':
    main()