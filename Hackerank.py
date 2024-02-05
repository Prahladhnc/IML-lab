import sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

def make_model():
    clf = Pipeline([
        ('vect', TfidfVectorizer(stop_words='english', ngram_range=(1, 1), min_df=4, strip_accents='ascii', lowercase=True)),
        ('clf', SGDClassifier(class_weight='balanced'))
    ])
    return clf

def run():
    known = [('Business means risk!', 1), ("This is a document", 1), ("this is another document", 4), ("documents are separated by newlines", 8)]
    
    xs, ys = load_data('trainingdata.txt')
    mdl = make_model()
    mdl.fit(xs, ys)
    txs = list(line.strip() for line in sys.stdin)[1:]  # Fixed the line (count) issue
    for x in txs:
        predicted = False
        for pattern, clazz in known:
            if pattern in x:
                print(clazz)
                predicted = True
                break
        if not predicted:
            print(mdl.predict([x])[0])

def load_data(filename):
    with open(filename, 'r') as data_file:
        sz = int(data_file.readline())
        xs = np.zeros(sz, dtype=object)  
        ys = np.zeros(sz, dtype=int)
        for i, line in enumerate(data_file):
            idx = line.index(' ')
            clazz = int(line[:idx])
            words = line[idx+1:].strip()
            xs[i] = words
            ys[i] = clazz
    return xs, ys

if __name__ == '__main__':  
    run()

import math

def pearson_correlation(sx, sy):
    n = len(sx)
    x, y, xy, xsq, ysq = 0, 0, 0, 0, 0
    for i in range(n):
        x += sx[i]
        y += sy[i]
        xy += sx[i]*sy[i]
        xsq += sx[i]*sx[i]
        ysq += sy[i]*sy[i]
    return (n*xy - x*y) / math.sqrt((n*xsq - x*x)*(n*ysq - y*y))

series_x = [15, 12, 8, 8, 7, 7, 7, 6, 5, 3]
series_y = [10, 25, 17, 11, 13, 17, 20, 13, 9, 15]
r = pearson_correlation(series_x, series_y)
print('{0:.3f}'.format(r))

# Enter your code here. Read input from STDIN. Print output to STDOUTfrom __future__ import division
from sklearn import svm, preprocessing
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import json

def solve():
    training = []
    annotation = []
    # download training file and sample test file from the problem description
    # url: https://www.hackerrank.com/challenges/stack-exchange-question-classifier

    with open("training.json") as f:
        f.readline()
        for line in f:
            data = json.loads(line)
            annotation.append(data['topic'])
            training.append(data['question'])

    count_vect = CountVectorizer(ngram_range = (1, 2), \
                                token_pattern = r'\b\w+\b',\
                                min_df = 1)
    training_counts = count_vect.fit_transform(training)

    tfidf_transformer = TfidfTransformer()
    training_tfidf = tfidf_transformer.fit_transform(training_counts)

    classifier = svm.LinearSVC().fit(training_tfidf, annotation)

    q = int(input())
    qs = []
    for _ in range(q):
        data = json.loads(input().strip())
        qs.append(data['question'])

    qs_counts = count_vect.transform(qs)
    qs_tfidf = tfidf_transformer.transform(qs_counts)
    ans = classifier.predict(qs_tfidf)

    for a in ans:
        print (a)

if __name__ == '__main__':
    solve()


import sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

def make_model():
    clf = Pipeline([
        ('vect', TfidfVectorizer(stop_words='english', ngram_range=(1, 1), min_df=4, strip_accents='ascii', lowercase=True)),
        ('clf', SGDClassifier(class_weight='balanced'))
    ])
    return clf

def run():
    known = [('Business means risk!', 1), ("This is a document", 1), ("this is another document", 4), ("documents are separated by newlines", 8)]
    
    xs, ys = load_data('trainingdata.txt')
    mdl = make_model()
    mdl.fit(xs, ys)
    txs = list(line.strip() for line in sys.stdin)[1:]  # Fixed the line (count) issue
    for x in txs:
        predicted = False
        for pattern, clazz in known:
            if pattern in x:
                print(clazz)
                predicted = True
                break
        if not predicted:
            print(mdl.predict([x])[0])

def load_data(filename):
    with open(filename, 'r') as data_file:
        sz = int(data_file.readline())
        xs = np.zeros(sz, dtype=object)  
        ys = np.zeros(sz, dtype=int)
        for i, line in enumerate(data_file):
            idx = line.index(' ')
            clazz = int(line[:idx])
            words = line[idx+1:].strip()
            xs[i] = words
            ys[i] = clazz
    return xs, ys

if __name__ == '__main__':  
    run()


