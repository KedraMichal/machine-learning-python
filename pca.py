import pandas as pd
import sklearn
from sklearn import decomposition, preprocessing
import numpy as np
np.set_printoptions(suppress=True)

df = pd.read_csv("cars.csv")

df = df[["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration","year"]]

data = df.to_numpy()
x = data[:,1:]
y = data[:,0]

scaler = preprocessing.StandardScaler()

data = scaler.fit_transform(df)

print(data)
pca = decomposition.PCA(n_components=4)

pca1 = pca.fit_transform(data)
print(pca1)


print(pca.explained_variance_ratio_)