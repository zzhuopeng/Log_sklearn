# -*- coding: utf-8 -*-
# sklearn基本用法2-通用学习模式
import numpy as np
from sklearn import datasets
# sklearn中有许多直接可用的数据库
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()  # iris表示一种花
iris_X = iris.data  # iris属性
iris_y = iris.target  # iris分类
# print(iris_X[:2, :])  # 4种属性
# print(iris_y)  # 3种分类

# 将数据集分为训练数据和测试数据（防止数据的相互影响）
# 并且会把原来排序的数据打乱（提高准确度）
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3)
# print(y_train)

# 使用sklearn中Classification方式
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)  # 开始训练
print('预测值', knn.predict(X_test))  # 预测测试数据
print('实际值', y_test)  # 与原数据进行对比

print('对比结果发现：大部分结果都预测对了，但是不可避免地存在某些误差')
