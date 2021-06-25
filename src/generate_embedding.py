import gensim
import numpy as np
import pandas as pd

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

if __name__ == '__main__':
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
