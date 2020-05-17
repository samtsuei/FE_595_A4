#1. Use a linear regression model with the Boston housing data set. Your code should then return which factor
#has the largest effect on the price of housing in Boston.
#(This is not the correlation coefficient. This is the absolute value of the slope.)
##reference  https://towardsdatascience.com/simple-and-multiple-linear-regression-in-python-c928425168f9

## import the data frame function


import pandas as pd

from sklearn import linear_model
from sklearn import datasets ## imports datasets from scikit-learn

def main():
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
 print('The largest effect on the price of housing in Boston is', header[maxpos], 'as of ', slope[maxpos])

if __name__ == "__main__":
 main()