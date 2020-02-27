import pandas as pd
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn import metrics
import matplotlib.pylab as plt

df = pd.read_csv("players_stats.csv", delimiter=";", decimal=",", index_col=0)
np.set_printoptions(suppress=True)

data = df.to_numpy()

data = preprocessing.scale(data)


def optimal_number_ofcluster(x_data, max_clusters):
    scores = []

    for i in range(max_clusters - 2):
        kmeans_met = KMeans(n_clusters=i + 2, n_init=20)
        kmeans_met = kmeans_met.fit(x_data)
        sil = metrics.silhouette_score(x_data, kmeans_met.labels_)
        # print("Number of clusters: {}, silhouette_score: {}".format(kmeans_met.n_clusters, sil))
        scores.append([sil, kmeans_met.inertia_])  # inertia - within-cluster sum of squares

    return scores


# results = optimal_number_ofcluster(data, 15)
# results = np.asarray(results, dtype=np.float32)
#
# plt.plot(np.arange(2, 15), results[:, 0])
# plt.xlabel("Number of clusters")
# plt.ylabel("Silhouette score")
# plt.show()
#
# plt.plot(np.arange(2, 15), results[:, 1])
# plt.xlabel("Number of clusters")
# plt.ylabel("Within-cluster sum of squares")
# plt.show()

kmeans = KMeans(n_clusters=5, n_init=20)
kmeans.fit(data)
# labels = np.array(kmeans.labels_)
# labels = labels.reshape(len(data), 1)
# arr = np.hstack([data, labels])

### Cluster statistics

df["cluster"] = kmeans.labels_
mean_stats = pd.DataFrame()
std_stats = pd.DataFrame()
for i in range(len(df.cluster.unique())):
    df_subset = df[df.cluster == i]
    df_subset = df_subset.drop(columns="cluster")
    mean_stats["Cluster {}".format(i)] = df_subset.mean()
    std_stats["Cluster {}-std".format(i)] = df_subset.std()

print(mean_stats)
print(std_stats)