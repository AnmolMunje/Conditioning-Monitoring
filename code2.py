import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


cols = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'label']
cls1 = pd.read_csv('train1.csv', names=cols, header=None).dropna()[:500]
cls2 = pd.read_csv('train2.csv', names=cols, header=None).dropna()
data = pd.concat([cls1, cls2], axis=0)
y = data['label']
X = data.drop(['label'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = svm.SVC()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print(accuracy_score(y_test, pred))