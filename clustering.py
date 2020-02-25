import pandas as pd
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn import preprocessing

data = pd.read_csv("players_stats.csv", delimiter=";", decimal=",", index_col=0)
np.set_printoptions(suppress=True)

data = data.to_numpy()

data = preprocessing.scale(data)
kmeans = KMeans(n_clusters=8, n_init= 20)

kmeans.fit(data)

print(kmeans.inertia_)
print(kmeans.labels_)
print(kmeans.cluster_centers_)
d = data[0,:]
print(kmeans.predict(d.reshape(1,7)))





