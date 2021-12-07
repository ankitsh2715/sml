import numpy as np
import pandas as pd
import matplotlib.pyplot as mplot


class K_Means:
    def __init__(self,num_components,data,initial_centroid=None):
        self.num_components = num_components
        self.data = data  
        self.initial_centroid = initial_centroid
        
    def setInitialCentroidVal(self,num_components,data):
        if(self.initial_centroid == 'random'): 
            initial_centroids = np.random.permutation(data.shape[0])[:self.num_components]
            self.centroids = data[initial_centroids]
        elif(self.initial_centroid == 'firstk'):
            self.centroids = data[:num_components]
        else:
            for i in range(self.num_components):
                self.centroids.append(i%self.num_components)
        return self.centroids    
 
    def fit(self,data):
        shp = np.shape(data)[0]
        centroid_val = self.setInitialCentroidVal(self.num_components,data)
        centroid_orig_val = centroid_val.copy()
        identify_cluster = np.mat(np.zeros((shp,2)))
        

        flag = True
        num_iter = 0
        while flag and num_iter<100:
            flag = False 
            
            for i in range(shp):
                min_dist = np.inf
                min_index = -1 
                
                for j in range(self.num_components):
                    dist_ji = calcEuclidDist(centroid_val[j,:],data[i,:])
                    if(dist_ji < min_dist):
                        min_dist = dist_ji
                        min_index = j 
                    
                    if identify_cluster[i, 0] != min_index: 
                        flag = True

                identify_cluster[i, :] = min_index, min_dist**2

            for cent in range(self.num_components):
                coord = data[np.nonzero(identify_cluster[:,0].A==cent)[0]]
                centroid_val[cent,:] = np.mean(coord, axis=0)
    
            num_iter += 1
            
        return centroid_val, identify_cluster, num_iter, centroid_orig_val


df=pd.read_csv(r'CSE575-HW03-Data.csv',header=None)
x=np.array(df)
df.isna()


def calcEuclidDist(x1, x2):
    return np.linalg.norm(x1-x2, axis=0)

def plot(index,centroids,orig_centroids):
    input = []
    colors = 10*["g","r","c","b","k"]

    for i in range(len(index)):
        for j in index[i]:
            input.append(int(j))
            
    j=0
    for i in input:
        mplot.scatter(x[j,0], x[j,1], marker="x", color=colors[i], s=150, linewidths=5)
        j+=1
    
    for cen in range(len(centroids)):
        mplot.scatter(centroids[cen][0],centroids[cen][1],marker="o", color="k", s=150, linewidths=5)
    
    for cen in range(len(orig_centroids)):
        mplot.scatter(orig_centroids[cen][0],orig_centroids[cen][1],marker="D", color="DarkBlue", s=150, linewidths=5)


def main():
    kmeans = K_Means(num_components=2,data = x,initial_centroid='random')
    centroids, cluster_assignments, iters, orig_centroids = kmeans.fit(x)
    index = cluster_assignments[:,0]
    plot(index,centroids,orig_centroids)

    x1=np.array(df)
    x1=df.iloc[0:,0:2]
    kmeans = K_Means(num_components=2,data = x1,initial_centroid='random')
    centroids, cluster_assignments, iters, orig_centroids = kmeans.fit(x)
    index = cluster_assignments[:,0]
    plot(index,centroids,orig_centroids)

if __name__ == "__main__":
    main()