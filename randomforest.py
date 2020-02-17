import pandas as pd
import seaborn as sns
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

data = pd.read_csv("heart_disease_dataset.csv")
print(data.head())
print(data.shape)

###  Missing values

print(data.isnull().sum())
data = data.dropna(how="any")


print(data.num.value_counts())  # 0-heart disease, 1- no heart disease
groupby_class = data.groupby("num")
print(groupby_class.describe())

#corr = data.corr()
#sns.heatmap(corr)
#plt.show()

data = data.drop(columns=["ca", "thal", "restecg", "trestbps", "fbs", "chol"])

data = data.to_numpy()

x = data[:, 0:6]
y = data[:, 7]

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

# dt_model = tree.DecisionTreeClassifier()
# dt_model.fit(x_train, y_train)
#
# rf_model = RandomForestClassifier()
# rf_model.fit(x_train, y_train)
#
# acc_dt = dt_model.score(x_test, y_test)
# acc_rf = rf_model.score(x_test, y_test)
# print(acc_dt, acc_rf)

### Comparison between random forest and decision tree accuaracy

kfold = StratifiedKFold(n_splits=10)
dt_model = tree.DecisionTreeClassifier()
rf_model = RandomForestClassifier()

cv_dt = cross_val_score(dt_model, x, y, scoring="accuracy", cv= kfold)
cv_rf = cross_val_score(rf_model, x, y, scoring="accuracy", cv= kfold)

print("Decision tree:{},\n Random forest:{}".format(cv_dt.mean(), cv_rf.mean()))
