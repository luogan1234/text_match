from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import argparse
import numpy as np
import copy
from scipy.sparse import vstack

def get_seqs(file):
    text = []
    with open('data/{}.tsv'.format(file), 'r') as f:
        for line in f:
            items = line.split('\t')
            text.append(items[0])
            text.append(items[1])
    return text

def save_feature(file, n, features):
    tmp = []
    for feature in features:
        shape = feature.shape
        if shape[0] == n:
            tmp.append(feature.reshape(n, shape[1]))
        else:
            tmp.append(feature.reshape(n, shape[1]*2))
    feature = np.concatenate(tmp, 1)
    print(file, 'feature shape:', feature.shape)
    np.save('data/{}.npy'.format(file), feature)

class BM25(object):
    def __init__(self, b=0.75, k1=1.6):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 4), norm=None, smooth_idf=False, token_pattern='\d+')
        self.b = b
        self.k1 = k1

    def fit(self, X):
        self.vectorizer.fit(X)
        count = super(TfidfVectorizer, self.vectorizer).transform(X)
        self.avdl = count.sum(1).mean()

    def transform(self, X):
        b, k1, avdl = self.b, self.k1, self.avdl
        count = super(TfidfVectorizer, self.vectorizer).transform(X)
        len_X = count.sum(1).A1
        w = k1*(1-b+b*len_X/avdl)
        idf = self.vectorizer._tfidf.idf_-1
        rows, cols = count.nonzero()
        for i, (row, col) in enumerate(zip(rows, cols)):
            v = count.data[i]
            count.data[i] = (k1+1)*v/(w[row]+v)*idf[col]
        return count

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Text_match')
    parser.add_argument('-train', type=str, default='train', choices=['train'])
    parser.add_argument('-test', type=str, default='testA', choices=['testA'])
    args = parser.parse_args()
    train = get_seqs(args.train)
    test = get_seqs(args.test)
    text = train+test
    
    tfidf = TfidfVectorizer(ngram_range=(1, 4), token_pattern='\d+')
    tfidf_feature = tfidf.fit_transform(text)
    print(tfidf_feature.shape)
    svd1 = TruncatedSVD(n_components=128).fit_transform(tfidf_feature)
    print(svd1.shape)
    
    bm25 = BM25()
    bm25.fit(text)
    bm25_feature = bm25.transform(text)
    print(bm25_feature.shape)
    svd2 = TruncatedSVD(n_components=128).fit_transform(bm25_feature)
    print(svd2.shape)
    
    count = CountVectorizer(ngram_range=(1, 1), token_pattern='\d+')
    count_feature = count.fit_transform(text)
    shape = count_feature.shape
    even_id = np.arange(0, shape[0], 2)
    count_even = count_feature[even_id]
    odd_id = np.arange(1, shape[0], 2)
    count_odd = count_feature[odd_id]
    count_d1 = count_odd-count_even
    count_d2 = count_even-count_odd
    count_d3 = copy.deepcopy(count_d1)
    count_d3[count_d3<0] = 0
    count_d4 = copy.deepcopy(count_d2)
    count_d4[count_d4<0] = 0
    count_delta = vstack([count_d1, count_d2])
    count_nonneg_d = vstack([count_d3, count_d4])
    count_abs = np.abs(count_odd-count_even)
    svd3 = TruncatedSVD(n_components=128).fit_transform(count_delta)
    svd4 = TruncatedSVD(n_components=128).fit_transform(count_nonneg_d)
    svd5 = TruncatedSVD(n_components=128).fit_transform(count_abs)
    d = shape[0]//2
    ids = np.concatenate([np.arange(d).reshape(d, 1), np.arange(d, d*2).reshape(d, 1)], 0).reshape(d*2)
    svd3 = svd3[ids]
    svd4 = svd4[ids]
    
    n1, n2, n3, n4 = len(train), len(test), len(train)//2, len(test)//2
    save_feature(args.train, n3, [svd1[:n1], svd2[:n1], svd3[:n1], svd4[:n1], svd5[:n3]])
    save_feature(args.test, n4, [svd1[-n2:], svd2[-n2:], svd3[-n2:], svd4[-n2:], svd5[-n4:]])