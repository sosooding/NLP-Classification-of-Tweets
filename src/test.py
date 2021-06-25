import gensim
import numpy as np

def tweet_vector(tokens, size):
	ret = np.zeros(size).reshape((1, size))
	count = 0
	for word in tokens:
		try:
			ret += model.wv[word].reshape((1, size))
			count += 1.0
		except KeyError:  # handling the case where the token is not in vocabulary
			continue
	if count != 0:
		ret /= count
	return ret

text = 'feeling shitty moment landed back hope one go sense agony loneliness'
model = gensim.models.Word2Vec.load("word2vec1.model")
print(len(tweet_vector(text.split(), 300)[0]))