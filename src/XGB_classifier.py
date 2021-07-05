from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

df = pd.read_csv('../data/50k_combined_embedded.csv')
X = df['W2V_Embedding'].tolist()

y = df['Classification']

for i in tqdm(range(len(X))):
    X[i] = X[i][1:len(X[i]) - 1].strip()
    X[i] = list(map(float, X[i].split()))
    X[i] = np.asarray(X[i])

X = np.asarray(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

xgb = XGBClassifier(max_depth=6, n_estimators=1000, nthread= 3).fit(X_train, y_train)
prediction = xgb.predict(X_test)
accuracy = accuracy_score(y_test, prediction)
print('Accuracy of XGB classifier on test set: {:.2f}'.format(
            accuracy))