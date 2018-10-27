from sklearn.naive_bayes import GaussianNB

import numpy as np

x = np.array([[11,141],[22,145],[293,20]])
y = np.array([188,245,123])

clf = GaussianNB()
clf.fit(x,y)

print(clf.score(x,y))
