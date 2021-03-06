from sklearn.linear_model import LogisticRegression
from sklearn import datasets, metrics

iris = datasets.load_iris()
classifier = LogisticRegression()
classifier.fit(iris.data, iris.target)
score = metrics.accuracy_score(iris.target, classifier.predict(iris.data))
print("Accuracy: %f" % score)