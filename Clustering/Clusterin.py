#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns

#importing the Iris dataset
dataset = pd.read_csv('iris.data')
data = dataset.iloc[:,:-1].to_numpy()
#print(data)

#____________________K-Means____________________
#Finding the optimum number of clusters for k-means classification
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)
    
#Plotting the results onto a line graph, allowing us to observe 'The elbow'
plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') #within cluster sum of squares
plt.show()

#Applying kmeans to the dataset / Creating the kmeans classifier
kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)

#Visualising the clusters
#Plotting the centroids of the clusters
plt.title('Clustering berdasarkan ukuran sepal')
plt.scatter(data[:,0], data[:,1])
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 150, c = 'pink')

#Visualising the clusters
#Plotting the centroids of the clusters
plt.title('Clustering berdasarkan ukuran petal')
plt.scatter(data[:,2], data[:,3])
plt.scatter(kmeans.cluster_centers_[:, 2], kmeans.cluster_centers_[:,3], s = 150, c = 'pink')

skor_kmeans = silhouette_score(data, y_kmeans)
print('akurasi K-Means : ', skor_kmeans)


#____________________Hierarchical____________________

H_predict = AgglomerativeClustering(n_clusters=3)
H_predict = H_predict.fit_predict(data)

plt.title('Clustering berdasarkan ukuran sepal')
plt.scatter(data[:,0], data[:,1], c=H_predict)

plt.title('Clustering berdasarkan ukuran petal')
plt.scatter(data[:,2], data[:,3], c=H_predict)

# generate the linkage matrix
Z = linkage(data, 'ward')

# set cut-off to 150
max_d = 7.08                # max_d as in max_distance

plt.figure(figsize=(25, 10))
plt.title('Iris Hierarchical Clustering Dendrogram')
plt.xlabel('Species')
plt.ylabel('distance')
dendrogram(
    Z,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=150,                  # Try changing values of p
    leaf_rotation=90.,      # rotates the x axis labels
    leaf_font_size=8.,      # font size for the x axis labels
)
plt.axhline(y=max_d, c='k')
plt.show()

skor_hierarchical = silhouette_score(data, H_predict)
print('akurasi Hierarchical : ', skor_hierarchical)