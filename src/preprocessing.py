import normalize_text
import pandas as pd

if __name__ == '__main__':
	f1 = open("../data/combined_lonely.txt", 'r')
	f2 = open("../data/SOLO-solitude-tweets.txt", 'r')
	
	l = []
	l2 = []
	classf = []

	cnt = 1

	for i in range(5000):

		print(cnt)

		l.append(f1.readline())
		classf.append("lonely")
		l2.append(normalize_text.normalize(l[-1]))
		cnt += 1

		print(cnt)

		l.append(f2.readline())
		classf.append("solitude")
		l2.append(normalize_text.normalize(l[-1]))
		cnt += 1

	dic = {'Tweet' : l, 'Cleaned' : l2, 'Classification' : classf}

	df = pd.DataFrame(dic)
	df.to_csv('../data/10k_combined_wsw.csv')