#1. Use a linear regression model with the Boston housing data set. Your code should then return which factor
#has the largest effect on the price of housing in Boston.
#(This is not the correlation coefficient. This is the absolute value of the slope.)
##reference  https://towardsdatascience.com/simple-and-multiple-linear-regression-in-python-c928425168f9

## import the data frame function
import pandas as pd

from sklearn import linear_model
from sklearn import datasets ## imports datasets from scikit-learn
data = datasets.load_boston() ## loads Boston dataset from datasets library

# define the data/predictors as the pre-set feature names
df = pd.DataFrame(data.data, columns=data.feature_names)

# Put the target (housing value -- MEDV) in another DataFrame
target = pd.DataFrame(data.target, columns=['MEDV'])
X = df
y = target['MEDV']

##run a multi varible linear regession
lm = linear_model.LinearRegression()
model = lm.fit(X,y)

##check the model fit
##print(lm.score(X,y))

slope = list(lm.coef_)
maxpos = slope.index(max(slope)) ## find hte position for the largest slope
header = list(X.head(0))
print('The largest effect on the price of housing in Boston is', header[maxpos])


#check the intecept
##print('Intercept',lm.intercept_)



#22. Use a KMeans regression model with the Iris data set.
# Graph the fit when using differing numbers of clusters.
# Graph the result and either corroborate or refute the assumption
# that the data set represents 3 different varieties of iris.

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

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
