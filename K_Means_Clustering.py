import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

x = np.array([[5, 3],
            [10, 15],
            [15, 12],
            [24, 10],
            [30, 45],
            [85, 70],
            [71, 80],])

# plt.scatter(x[:, 0], x[:, 1])
# plt.show()

km = KMeans(n_clusters=2)
km.fit(x)

print(f"K map cluster is {km.cluster_centers_}")
print(f"K map level is {km.labels_}")

km = KMeans(n_clusters=3)
km.fit(x)
plt.scatter(x[:, 0], x[:, 1], c=km.labels_, cmap='rainbow')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],  color ='black')
plt.show()

