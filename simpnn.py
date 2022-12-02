from sklearn.linear_model import Perceptron
import numpy as np
from sklearn.datasets import load_iris
iris = load_iris()
x = iris.data[:,(2,3)]
y = (iris.target == 0).astype(int)

per_clf = Perceptron()
per_clf.fit(x,y)

y_pred = per_clf.predict([[2,0.5]])

print(y_pred)