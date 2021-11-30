from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

import numpy as np
import csv
from sklearn.utils import Bunch

def load_crime_dataset():
    with open(r'10_Property_stolen_and_recovered.csv') as csv_file:
        data_reader = csv.reader(csv_file)
        feature_names = next(data_reader)[:-1]
        data = []
        target = []
        for row in data_reader:
            if(row != []):
                features = row[4:]
                label = row[3][0]
                data.append([float(num) for num in features])
                target.append(int(label))
        
        data = np.array(data)
        target = np.array(target)
    return Bunch(data=data, target=target, feature_names=feature_names)

data = load_crime_dataset()
classifier = DecisionTreeClassifier()
classifier.fit(data.data, data.target)
score = metrics.accuracy_score(data.target, classifier.predict(data.data))
print("Accuracy: %f" % score)