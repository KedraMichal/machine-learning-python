import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.csv")


for col in data:
    print(data[col].unique())

# data.buying[data.buying=="vhigh"] = 0
# data.buying[data.buying=="high"] = 1
# data.buying[data.buying=="med"] = 2
# data.buying[data.buying=="low"] = 3

le = preprocessing.LabelEncoder()
buying = le.fit_transform(data["buying"])
maint = le.fit_transform(data["maint"])
door = le.fit_transform(data["door"])
persons = le.fit_transform(data["persons"])
lug_boot = le.fit_transform(data["lug_boot"])
safety = le.fit_transform(data["safety"])
obj_class = le.fit_transform(data["class"])


arr = np.vstack([buying, maint, door, persons, lug_boot, safety, obj_class])
arr = arr.transpose()
x = arr[:, 0:6]
y = arr[:, 6]

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

knn_model = KNeighborsClassifier(n_neighbors=3).fit(x_train, y_train)

acc = knn_model.score(x_test, y_test)
print(acc)