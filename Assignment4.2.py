#2. Use a KMeans regression model with the Iris data set.
# Graph the fit when using differing numbers of clusters.
# Graph the result and either corroborate or refute the assumption
# that the data set represents 3 different varieties of iris.

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import datasets ## imports datasets from scikit-learn
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

def main():
 df = datasets.load_iris()
 x=df.data

##use Elbow method to Graph the result and
##either corroborate or refute the assumption that the data set represents 3 different varieties of iris

 kmeans5 = KMeans(n_clusters=5)
 y_kmeans5 = kmeans5.fit_predict(x)
#print(y_kmeans5)
 plt.scatter(x[:,0],x[:,1],c=y_kmeans5, cmap='rainbow')
 plt.title('Cluster=5')
 plt.show()

 kmeans3 = KMeans(n_clusters=3)
 y_kmeans3 = kmeans3.fit_predict(x)
#print(y_kmeans3)
 plt.title('Cluster=3')
 plt.scatter(x[:, 0], x[:, 1], c=y_kmeans3, cmap='rainbow')
 plt.show()

 Error =[]
 for i in range(1, 11):
    kmeans = KMeans(n_clusters = i).fit(x)
    kmeans.fit(x)
    Error.append(kmeans.inertia_)

 plt.plot(range(1, 11), Error)
 plt.title('Elbow method')
 plt.xlabel('No of clusters')
 plt.ylabel('Error')
 plt.show()

if __name__ == "__main__":
 main()