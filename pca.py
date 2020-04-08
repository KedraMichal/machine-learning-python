import pandas as pd
import sklearn
from sklearn import decomposition, preprocessing
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)
df = pd.read_csv("cars.csv")

df = df[["origin", "mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "year"]]

data = df.to_numpy()
x = data[:, 1:]
y = data[:, 0]

scaler = preprocessing.StandardScaler()
data = scaler.fit_transform(x)

pca = decomposition.PCA(n_components=4)
pca1 = pca.fit_transform(data)

print(pca1)
print(pca.explained_variance_ratio_)

plt.scatter(pca1[:, 0], pca1[:, 1])
plt.show()


# with pipeline
pipe = Pipeline([('scaler', preprocessing.StandardScaler()),
                 ('reducer', decomposition.PCA(n_components=4)),
                 ('knn', KNeighborsClassifier())])

pipe.fit(x, y)
print(pipe.steps[1][1].explained_variance_ratio_)
plt.plot(pipe.steps[1][1].explained_variance_ratio_)

plt.xlabel('Principal component index')
plt.ylabel('Explained variance ratio')
plt.show()

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1, random_state=0)

### Choosing optimal hyperparameter
parameters = {'knn__n_neighbors': [x for x in range(1, 20)]}
grid = GridSearchCV(pipe, param_grid=parameters, cv=5)
grid.fit(x_train, y_train)
print(grid.best_params_)
print(grid.best_score_)

knn = KNeighborsClassifier(n_neighbors=4).fit(x_train, y_train)
print(knn.score(x_test, y_test))
