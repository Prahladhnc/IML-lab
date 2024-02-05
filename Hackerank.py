# 1 Stock prediction

from __future__ import division
from math import sqrt
from operator import add
from heapq import heappush, heappop

def printTransactions(money, k, d, name, owned, prices):
    def mean(nums):
        return sum(nums) / len(nums)

    def sd(nums):
        average = mean(nums)
        return sqrt(sum([(x - average) ** 2 for x in nums]) / len(nums))

    def info(price):
        cc, sigma, acc = 0, 0.0, 0
        for i in range(1, 5): 
            if price[i] > price[i - 1]: cc += 1
        sigma = sd(price)
        mu = mean(price)
        c1, c2, c3 = mean(price[0:3]), mean(price[1:4]), mean(price[2:5])
        
        return (price[-1] - price[-2]) / price[-2]
    
    infos = map(info, prices)
    res = []
    
    drop = []
    
    for i in range(k):
        cur_info = info(prices[i])
        if cur_info > 0 and owned[i] > 0:
            res.append((name[i], 'SELL', str(owned[i])))
        elif cur_info < 0:
            heappush(drop, (cur_info, i, name[i]))
    
    while money > 0.0 and drop:
        rate, idx, n = heappop(drop)
        amount = int(money / prices[idx][-1])
        if amount  > 0:
            res.append((n, 'BUY', str(amount)))
            money -= amount * prices[idx][-1]
    
    print len(res)
    for r in res:
        print ' '.join(r)
    
    

if __name__ == '__main__':
    m, k, d = [float(i) for i in raw_input().strip().split()]
    k = int(k)
    d = int(d)
    names = []
    owned = []
    prices = []
    for data in range(k):
        temp = raw_input().strip().split()
        names.append(temp[0])
        owned.append(int(temp[1]))
        prices.append([float(i) for i in temp[2:7]])

    printTransactions(m, k, d, names, owned, prices)



#2)Stack Exchange
from __future__ import division
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

    q = int(raw_input())
    qs = []
    for _ in range(q):
        data = json.loads(raw_input().strip())
        qs.append(data['question'])

    qs_counts = count_vect.transform(qs)
    qs_tfidf = tfidf_transformer.transform(qs_counts)
    ans = classifier.predict(qs_tfidf)

    for a in ans:
        print a

if __name__ == '__main__':
    solve()


# 3) Document Classification

from __future__ import division
from sklearn import svm, preprocessing
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

def solve():
    training = []
    annotation = []
    # download from the problem description page and place it in the same dir
    # url: https://www.hackerrank.com/challenges/document-classification
    with open("trainingdata.txt") as f:
        f.readline()
        for line in f:
            data = line.split()
            annotation.append(int(data[0]))
            training.append(' '.join(data[1:]))

    count_vect = CountVectorizer(ngram_range = (1, 3), \
                                token_pattern = r'\b\w+\b',\
                                min_df = 4)
    training_counts = count_vect.fit_transform(training)

    tfidf_transformer = TfidfTransformer()
    training_tfidf = tfidf_transformer.fit_transform(training_counts)

    classifier = svm.LinearSVC().fit(training_tfidf, annotation)

    q = int(raw_input())
    qs = []
    for _ in range(q):
        qs.append(raw_input().strip())

    qs_counts = count_vect.transform(qs)
    qs_tfidf = tfidf_transformer.transform(qs_counts)
    ans = classifier.predict(qs_tfidf)

    for a in ans:
        print a

if __name__ == '__main__':
    solve()


#4) Pearson Coefficient

import math as m

# Define functions
def pearson(first_data, second_data, n):
    # Numerator part
    sum_firt_data       = sum(first_data)
    sum_second_data     = sum(second_data)
    sum_data            = sum([x*y for x,y in zip(first_data, second_data)])

    # Denominator part
    sum_first_data_squared      = sum([x**2 for x in first_data])
    sum_first_data_mult_squared = sum_firt_data ** 2
    sum_secon_data_squared      = sum([y**2 for y in second_data])
    sum_secon_data_mult_squared = sum_second_data ** 2

    numerator       = (n * sum_data) - (sum_firt_data * sum_second_data)
    den_first_data  = m.sqrt((n * sum_first_data_squared) - sum_first_data_mult_squared)
    den_second_data = m.sqrt((n * sum_secon_data_squared) - sum_secon_data_mult_squared)

    return round(numerator / (den_first_data * den_second_data), 2)


# Set data
n = int(input())
mathematics = []
physics     = []
chemistry   = []
for i in range(n):
    elements = list(map(float, input().split()))
    mathematics.append(elements[0])
    physics.append(elements[1])
    chemistry.append(elements[2])

# Show the correlation
print (pearson(mathematics, physics, float(n)))
print (pearson(physics, chemistry, float(n)))
print (pearson(mathematics, chemistry, float(n))
