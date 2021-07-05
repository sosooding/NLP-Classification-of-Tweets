import pandas as pd
import gensim 
import numpy as np
from gensim.models.doc2vec import TaggedDocument
from tqdm import tqdm

def add_label(tweets):

    ret = []
    for i in range(len(tweets)):
        ret.append(TaggedDocument(tweets[i], [i]))
    return ret

#train = pd.read_csv("../data/train_combined.csv")
combined = pd.read_csv("../data/50k_combined.csv")

#combined = train.append(test, ignore_index = True, sort = True)
tokenized_tweet = combined['Cleaned'].apply(lambda x: x.split())

labeled_tweets = add_label(tokenized_tweet)

model_d2v = gensim.models.doc2vec.Doc2Vec(dm=1, # dm = 1 for ‘distributed memory’ model
                                  dm_mean=1, # dm_mean = 1 for using mean of the context word vectors
                                  vector_size=300, # no. of desired features
                                  window=5, # width of the context window                                  
                                  negative=7, # if > 0 then negative sampling will be used
                                  min_count=5, # Ignores all words with total frequency lower than 5.                                  
                                  workers=32, # no. of cores                                  
                                  alpha=0.1, # learning rate                                  
                                  seed = 23, # for reproducibility
                                 ) 

model_d2v.build_vocab([i for i in tqdm(labeled_tweets)])
model_d2v.train(labeled_tweets, total_examples= len(combined['Cleaned']), epochs=50)

model_d2v.save("../models/doc2vec1.model")

model_d2v = gensim.models.Doc2Vec.load("../models/doc2vec1.model")

l = []
for i in range(len(combined)):
    l.append(model_d2v.infer_vector(tok enized_tweet))

combined['Doc2Vec_Embedded'] = l
combined.to_csv('../data/50k_combined_embedded2.csv')