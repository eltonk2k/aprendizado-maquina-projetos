from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.linear_model import SGDRegressor
import numpy as np

def train_perceptron(X_train, y_train):
    clf = Perceptron(max_iter=1000)
    clf.fit(X_train, y_train)
    return clf

def train_linear_regression(X_train, y_train):
    clf = SGDRegressor(max_iter=1000)
    clf.fit(X_train, y_train)
    return clf

def train_logistic_regression(X_train, y_train):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    return clf

def predict(clf, X_test):
    return clf.predict(X_test)
