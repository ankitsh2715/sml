import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

df = pd.read_csv(r'CSE575-HW03-Data.csv', header=None)

x = np.array(df)

df.isna


class KMeansClustering():
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.centroids = None

    def fit_transform(self, X, n_iter=20):
        size = X.shape[0]
        xmax = X.max(dim=0)[0].unsqueeze(1)
        xmin = X.min(dim=0)[0].unsqueeze(1)

        dists = torch.zeros((size, self.n_clusters))
        best_loss = 1e10
        pred = None

        for _ in range(n_iter):
            centroids = (xmin - xmax) * torch.rand((X.shape[1], self.n_clusters)) + xmax
            old_loss = -1
            while 1:
                for i in range(self.n_clusters):
                    ctr = centroids[:, i].unsqueeze(1)
                    dists[:, i] = (X - ctr.T).pow(2).sum(dim=1).sqrt()
                dists_min, labels = dists.min(dim=1)

                for i in range(self.n_clusters):
                    idx = torch.where(labels == i)[0]
                    if len(idx) == 0:
                        continue
                    centroids[:, i] = X[idx].mean(dim=0)

                new_loss = dists_min.sum()
                if old_loss == new_loss:
                    break
                old_loss = new_loss
            if new_loss < best_loss:
                best_loss = new_loss
                pred = labels
        return pred, dists_min


X = torch.from_numpy(x).float()

objectiveCostValues1 = []
objectiveCostValues2 = []
objectiveCostValues3 = []

kValues = []


def k_means(k):
    kms = KMeansClustering(n_clusters=k)
    pred, dists_min = kms.fit_transform(X)

    objectiveCostValues1.append(torch.sum(dists_min))
    objectiveCostValues2.append(torch.sum(dists_min ** 2))
    objectiveCostValues3.append(math.sqrt(torch.sum(dists_min ** 2)))
    kValues.append(k)


for k in [2, 3, 4, 5, 6, 7, 8, 9]:
    k_means(k)
plt.figure('1')
plt.plot(kValues, objectiveCostValues1)
plt.xlabel("Values of K")
plt.ylabel("Objective Cost (sum of distance)")
plt.figure('2')
plt.plot(kValues, objectiveCostValues2)
plt.xlabel("Values of K")
plt.ylabel("Objective Cost (sum of square of distance)")
plt.figure('3')
plt.plot(kValues, objectiveCostValues3)
plt.xlabel("Values of K")
plt.ylabel("Objective Cost (sqrt of sum of square of distance)")
plt.figure('0')

x_2 = df.iloc[0:, 0:2]
X = torch.from_numpy(np.array(x_2)).float()

kms = KMeansClustering(n_clusters=2)
pred, dists_min = kms.fit_transform(X)

plt.scatter(X[:, 0], X[:, 1], c=pred)