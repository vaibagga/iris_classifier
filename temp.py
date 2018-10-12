'''
This program trains a classifier and predicts species
for the famous iris dataset (which is a good dataset
for begginers in data science) using logistic regression in
scikit learn library in Python

Scikit learn has a function for loading the iris dataset
The dataset can bve imported in CSV format at 
https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv
'''

from sklearn.datasets import load_iris
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Load the iris dataset using load_iris function in scikit learn library

iris_dataset = load_iris()
log = LogisticRegression()

'''
The target of iris dataset contains four parameters:
    sepal_length,sepal_width,petal_length and petal_width
The data consists of species
'''

y = iris_dataset.target
X = iris_dataset.data

'''
For purpose of training and testing data is split
into 2 proportions of 70 and 30 per cent respectively
'''

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3) 

'''
Use only training data for training the linear regression model 
Since the dataset is small it can be quickly retrained and 
there is no need of creating a pickle file
'''

log.fit(X_train,y_train)

# Predict the value of species for testing data set

y_predict = log.predict(X_test)

# print the accuracy of the data set for testing data set

print(accuracy_score(y_test,y_predict))
