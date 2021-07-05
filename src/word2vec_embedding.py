import pandas as pd
import gensim 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def tweet_vector(tokens, size):
    ret = np.zeros(size).reshape((1, size))
    count = 0
    for word in tokens:
        try:
            ret += model.wv[word].reshape((1, size))
            count += 1.0
        except KeyError:
            continue
    if count != 0:
        ret /= count
    return ret

#Training the model
combined = pd.read_csv("../data/50k_combined.csv")
'''
print("Training")
tokenized_tweet = combined['Cleaned'].apply(lambda x: x.split())
model_w2v = gensim.models.Word2Vec(
            tokenized_tweet,
            vector_size = 300,
            window = 5,
            sg = 1,
            hs = 0,
            negative = 10,
            workers= 32,
            seed = 34
)

model_w2v.train(tokenized_tweet, total_examples= len(combined['Cleaned']), epochs=50)
model_w2v.save("../models/word2vec1.model")
'''
print("Trained")

# Applying the trained model
model = gensim.models.Word2Vec.load("../models/word2vec1.model")
df1 = pd.read_csv('../data/50k_combined.csv')
X = []
cnt = 1
for i in df1['Cleaned']:
    vec = tweet_vector(i.split(), 300)[0]
    X.append(vec)
    cnt += 1

print("Embedded")


df1['W2V_Embedding'] = X
df1.to_csv('../data/50k_combined_embedded.csv')
