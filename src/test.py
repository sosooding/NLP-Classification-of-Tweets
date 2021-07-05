import pandas as pd
import gensim 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from normalize_text import normalize

def tweet_vector(tokens, size):
    ret = np.zeros(size).reshape((1, size))
    count = 0
    for word in tokens:
        try:
            ret += model1.wv[word].reshape((1, size))
            count += 1.0
        except KeyError:
            continue
    if count != 0:
        ret /= count
    return ret

s1 = 'i am happy'
s2 = 'i am unhappy'
s3 = 'i am not happy'
model1 = gensim.models.Word2Vec.load("../models/word2vec1.model")
model2 = gensim.models.Doc2Vec.load("../models/doc2vec1.model")
arr1 = [np.asarray(tweet_vector(normalize(s1), 300))[0]]
arr2 = [np.asarray(tweet_vector(normalize(s2), 300))[0]]
arr3 = [np.asarray(tweet_vector(normalize(s3), 300))[0]]
print(cosine_similarity(arr1, arr2))
print(cosine_similarity(arr1, arr3))
print(cosine_similarity(arr2, arr3))