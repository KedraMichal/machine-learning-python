import pandas as pd
import numpy as np
import sklearn
from sklearn import model_selection
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

data = pd.read_csv('cars.csv')
np.set_printoptions(suppress=True)

data = data[['mpg', 'displacement', 'weight', 'cylinders', 'origin']]

arr = data.to_numpy()
x = arr[:, 0:4]
y = arr[:, 4]

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

svm_model = svm.SVC(kernel='linear', C=3)
svm_model.fit(x_train, y_train)

predicted = svm_model.predict(x_test)
conf_mat = confusion_matrix(y_test, predicted)
print(conf_mat)

svm_linear = svm.SVC(kernel='linear', C=3)
# svm_model3.fit(x_train, y_train)

svm_poly = svm.SVC(kernel='poly', C=3)
# svm_model2.fit(x_train, y_train)

# acc = svm_model.score(x_test, y_test)
# acc2 = svm_model2.score(x_test, y_test)

### K-fold ross validation
cv_results = cross_val_score(svm_linear, x_train, y_train, cv=10, scoring='accuracy')
cv_results2 = cross_val_score(svm_poly, x_train, y_train, cv=10, scoring='accuracy')

print("Linear:{}, poly:{}".format(cv_results.mean(), cv_results2.mean()))

### Stratified k-fold cv - each fold contains roughly the same proportions of the x (here 3) types of class labels

strat = StratifiedKFold(n_splits=10, shuffle=True, random_state=3)
strat_results = cross_val_score(svm_linear, x_train, y_train, cv=strat, scoring='accuracy')
strat_results2 = cross_val_score(svm_poly, x_train, y_train, cv=strat, scoring='accuracy')

print("Linear:{}, poly:{}".format(strat_results.mean(), strat_results2.mean()))