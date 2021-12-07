import numpy as np
import pandas as pd
import matplotlib.pyplot as mplot
import seaborn as sb
from scipy.stats import multivariate_normal


class GMM:
    def __init__(self, num_components, max_iterations=5):
        self.num_components = num_components
        self.max_iterations = int(max_iterations)
    
    def initialize(self, V):
        self.shape = V.shape
        self.r, self.c = self.shape

        self.phi = np.full(shape=self.num_components, fill_value=1 / self.num_components)
        self.wgt = np.full(shape=self.shape, fill_value=1 / self.num_components)

        rand_partition = np.random.randint(low=0, high=self.r, size=self.num_components)
        self.mu = [V[row_index, :] for row_index in rand_partition]
        self.sigma = [np.cov(V.T) for _ in range(self.num_components)]


    def fit(self, V):
        self.initialize(V)
        for i in range(self.max_iterations):
            self.EStep(V)
            self.MStep(V)


    def predictWrapper(self, V):
        p = np.zeros((self.r, self.num_components))

        for i in range(self.num_components):
            dist = multivariate_normal(mean=self.mu[i], cov=self.sigma[i])
            p[:, i] = dist.pdf(V)

        temp = p * self.phi
        return temp / (temp.sum(axis=1)[:, np.newaxis])


    def predict(self, V):
        return np.argmax(self.predictWrapper(V), axis=1)


    def EStep(self, V):
        self.wgt = self.predictWrapper(V)
        self.phi = self.wgt.mean(axis=0)


    def MStep(self, V):
        for i in range(self.num_components):
            w = self.wgt[:, [i]]
            total = w.sum()
            self.mu[i] = (V * w).sum(axis=0) / total
            self.sigma[i] = np.cov(V.T, aweights=(w / total).flatten(), bias=True)

def main():
    df = pd.read_csv(r"CSE575-HW03-Data.csv", header=None)
    X = np.array(df)

    gmm = GMM(num_components=2, max_iterations=15)
    gmm.fit(X)
    gmm.predict(X)

    mplot.figure(figsize=(7.5, 7.5))
    sb.scatterplot(data=df, x=0, y=1, hue=gmm.predict(X), palette=["red", "blue"])

if __name__ == "__main__":
    main()