from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import argparse
import numpy as np

def get_seqs(file):
    text = []
    with open('data/{}.tsv'.format(file), 'r') as f:
        for line in f:
            items = line.split('\t')
            text.append(items[0])
            text.append(items[1])
    return text

def save_feature(file, feature1, feature2):
    shape = feature1.shape
    feature1 = feature1.reshape(shape[0]//2, shape[1]*2)
    shape = feature2.shape
    feature2 = feature2.reshape(shape[0]//2, shape[1]*2)
    feature = np.concatenate([feature1, feature2], 1)
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
    svd_feature1 = TruncatedSVD(n_components=128).fit_transform(tfidf_feature)
    print(svd_feature1.shape)
    bm25 = BM25()
    bm25.fit(text)
    bm25_feature = bm25.transform(text)
    print(bm25_feature.shape)
    svd_feature2 = TruncatedSVD(n_components=128).fit_transform(bm25_feature)
    print(svd_feature2.shape)
    save_feature(args.train, svd_feature1[:len(train)], svd_feature2[:len(train)])
    save_feature(args.test, svd_feature1[-len(test):], svd_feature2[-len(test):])