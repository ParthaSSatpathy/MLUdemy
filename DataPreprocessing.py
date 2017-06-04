# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
path = 'C:/Users/parth/Desktop/Udemy/ML/Machine Learning A-Z Template Folder/Part 1 - Data Preprocessing/'
dataset = pd.read_csv(path+'Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

##Splitting train and test data set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 25)

# Feature Scaling
## Consider the Age and Salary here. In ML when we try to find distance, we d using Eucladian Distance
## For Age the range is very small however for Salary here the rnge is very big
## So we have to make them to similar range, normally betw-1 to 1
## We do that using two options: Standardization and Normalization
## Stabdardization: x = (x-mean(x))/sd(x)
## Normalization: x = (x-min(x))/(max(x)-min(x))

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)