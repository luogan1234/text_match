import numpy as np
import argparse
import os
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Text_match')
    parser.add_argument('-train', type=str, default='train', choices=['train'])
    parser.add_argument('-test', type=str, default='testA', choices=['testA'])
    args = parser.parse_args()
    labels = []
    with open('data/{}.tsv'.format(args.train), 'r') as f:
        for line in f:
            items = line.split('\t')
            labels.append(int(items[2]))
    train_f0 = np.load('result/features/{}.npy'.format(args.train))
    test_f0 = np.load('result/features/{}.npy'.format(args.test))
    path = '.' #'/data/luogan/text_match'
    predicts = []
    for seed in range(100):
        file = '{}/result/features/{}_{}_{}.npy'.format(path, args.train, args.test, seed)
        if not os.path.exists(file):
            continue
        print('Ensemble id:', seed)
        features = np.load(file)
        train_f, test_f = features[:len(labels)], features[len(labels):]
        model = LogisticRegression(tol=1e-3, n_jobs=5, C=0.01, max_iter=100, multi_class='ovr')
        X = np.concatenate([train_f0, train_f], 1)
        #X = train_f
        model.fit(X, labels)
        X = np.concatenate([test_f0, test_f], 1)
        #X = test_f
        y = model.predict_proba(X)
        predicts.append(y[:, 1])
    predicts = np.mean(np.array(predicts), 0).tolist()
    with open('result/result.csv', 'w') as f:
        f.write('\n'.join([str(v) for v in predicts]))