# -*- coding:utf-8 -*
from numpy import hstack, vstack, array, median, nan
from numpy.random import choice
from sklearn.datasets import load_iris


iris = load_iris()
iris.data = hstack((choice([0, 1, 2], size=iris.data.shape[0]+1).reshape(-1,1), vstack((iris.data, array([nan, nan, nan, nan]).reshape(1,-1)))))

iris.target = hstack((iris.target, array([median(iris.target)])))
print(iris.target)
