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
    parser = argparse.ArgumentParser(description='Text_match')
    parser.add_argument('-train', type=str, default='train', choices=['train'])
    parser.add_argument('-test', type=str, default='testA', choices=['testA'])
    parser.add_argument('-lr0', type=float, default=3e-4)
    parser.add_argument('-lr', type=float, default=6e-5)
    parser.add_argument('-bs', type=int, default=32)
    parser.add_argument('-mask_w', type=float, default=0)
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-cpu', action='store_true')
    args = parser.parse_args()
    set_seed(args.seed)
    config = Config(args.train, args.test, args.lr0, args.lr, args.bs, args.mask_w, args.seed, args.cpu)
    data_loader = MyDataLoader(config)
    processor = Processor(data_loader, config)
    processor.train()
    #processor.extract_feature()
    #processor.predict()

if __name__ == '__main__':
    main()