import pandas as pd
import gensim 
import numpy as np

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
train = pd.read_csv("../data/train_combined.csv")
test = pd.read_csv("../data/test_combined.csv")

combined = train.append(test, ignore_index = True, sort = True)

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

# Applying the trained model
model = gensim.models.Word2Vec.load("../models/word2vec1.model")
df = pd.read_csv('../data/test_combined.csv')
l = []
cnt = 1
for i in df['Cleaned']:
    vec = tweet_vector(i.split(), 300)
    l.append(vec)
    cnt += 1

df['W2V_Embedding'] = l
df.to_csv('../data/test_combined_embedded.csv')