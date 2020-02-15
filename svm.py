import pandas as pd
import numpy as np
import sklearn
from sklearn import model_selection
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score



data = pd.read_csv('cars.csv')
np.set_printoptions(suppress=True)

data = data[['mpg', 'displacement', 'weight', 'cylinders', 'origin']]

arr = data.to_numpy()
x = arr[:, 0:4]
y = arr[:, 4]

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

svm_model = svm.SVC(kernel='linear', C=3)
#svm_model.fit(x_train, y_train)

svm_model2 = svm.SVC(kernel='poly', C=3)
# svm_model2.fit(x_train, y_train)

#acc = svm_model.score(x_test, y_test)
#acc2 = svm_model2.score(x_test, y_test)

kfold = StratifiedKFold(n_splits=7)

cv_results = cross_val_score(svm_model, x_train, y_train, cv=kfold)
cv_results2 = cross_val_score(svm_model2, x_train, y_train, cv=kfold)

print("Linear:{}, poly:{}".format(cv_results.mean(), cv_results2.mean()))



