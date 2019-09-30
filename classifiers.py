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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.ensemble import RandomForestClassifier

#Load the iris dataset using load_iris function in scikit learn library
iris_dataset = load_iris()

'''
The target of iris dataset contains four parameters:
    sepal_length,sepal_width,petal_length and petal_width
The data consists of species
'''

y = iris_dataset.target
X = iris_dataset.data

'''
For purpose of training and testing data is split
into 2 proportions of 80 and 20 per cent respectively
'''

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1000)

'''
4 different classifiers' initialization
'''
GNB_clf= GaussianNB()
KNN_clf = KNeighborsClassifier(n_neighbors=3)
DT_clf = DecisionTreeClassifier(random_state=100)
SVM_clf = SVC(random_state=100)
RF_clf = RandomForestClassifier(max_features = 4, random_state=100)

'''
Use only training data for training the 4 different models 
Since the dataset is small it can be quickly retrained and 
there is no need of creating a pickle file
'''
GNB_clf.fit(X_train, y_train)
KNN_clf.fit(X_train, y_train)
DT_clf.fit(X_train, y_train)
SVM_clf.fit(X_train, y_train)
RF_clf.fit(X_train, y_train)

# Predict the value of species for testing data set
GNB_pred = GNB_clf.predict(X_test)
KNN_pred = KNN_clf.predict(X_test)
DT_pred = DT_clf.predict(X_test)
SVM_pred = SVM_clf.predict(X_test)
RF_Pred = RF_clf.predict(X_test)

# print the accuracy of the data set for testing data set
print('Naive Bayes: ',round(accuracy_score(y_test, GNB_pred), 2))
print('KNN: ', round(accuracy_score(y_test, KNN_pred), 2))
print('Decision Trees: ',round(accuracy_score(y_test, DT_pred), 2))
print('Support Vector Machines: ', round(accuracy_score(y_test, SVM_pred), 2))
print('Random Forest: ', round(accuracy_score(y_test, RF_Pred), 2))
