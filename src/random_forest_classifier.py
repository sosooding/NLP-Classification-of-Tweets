from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tqdm import tqdm

'''
For using Word2Vec Embedding

df = pd.read_csv('../data/50k_combined_embedded.csv')
X = df['W2V_Embedding'].tolist()

'''


# For using Doc2Vec Embedding

df = pd.read_csv('../data/50k_combined_embedded2.csv')
X = df['Doc2Vec_Embedded'].tolist()

y = df['Classification']

for i in tqdm(range(len(X))):
    X[i] = X[i][1:len(X[i]) - 1].strip()
    X[i] = list(map(float, X[i].split()))
    X[i] = np.asarray(X[i])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
rf = RandomForestClassifier(n_estimators=1000, criterion='entropy')
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print('Accuracy of random forest classifier on test set: {:.2f}'.format(
            rf.score(X_test, y_test)))
