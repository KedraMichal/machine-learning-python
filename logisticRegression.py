import numpy as np
import pandas as pd
import sklearn
from sklearn import model_selection
from sklearn import linear_model
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.metrics import classification_report

data = pd.read_csv("heart_disease_dataset.csv")


# print(data.num.value_counts())
# groupby_class = data.groupby("num")
# print(groupby_class)
data = data.dropna(how="any")
data = data.drop(columns=["ca", "thal", "restecg", "trestbps", "fbs", "chol"])
data = data.to_numpy()

x = data[:, 0:6]
y = data[:, 7]


x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

logistic_r = linear_model.LogisticRegression()
logistic_r.fit(x_train, y_train)

predicted = logistic_r.predict(x_test)

actual_vs_predicted = [(predicted[i], y_test[i]) for i in range(len(predicted))]
print(actual_vs_predicted)

acc = logistic_r.score(x_test, y_test)
print(acc)

conf = confusion_matrix(y_test, predicted)
print(conf)

class_report = classification_report(y_test, predicted)
print(class_report)