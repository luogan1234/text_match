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

def save_feature(file, feature):
    shape = feature.shape
    print(file, 'feature shape:', shape)
    feature = feature.reshape(shape[0]//2, 2, shape[1])
    np.save('data/{}.npy'.format(file), feature)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Text_match')
    parser.add_argument('-train', type=str, default='train', choices=['train'])
    parser.add_argument('-test', type=str, default='testA', choices=['testA'])
    args = parser.parse_args()
    train = get_seqs(args.train)
    test = get_seqs(args.test)
    text = train+test
    tfidf = TfidfVectorizer(ngram_range=(1, 5))
    tfidf_feature = tfidf.fit_transform(text)
    print(tfidf_feature.shape)
    svd_feature = TruncatedSVD(n_components=128).fit_transform(tfidf_feature)
    print(svd_feature.shape)
    save_feature(args.train, svd_feature[:len(train)])
    save_feature(args.test, svd_feature[-len(test):])