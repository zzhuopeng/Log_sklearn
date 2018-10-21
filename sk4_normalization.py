# -*- coding: utf-8 -*-
from sklearn import preprocessing  # 数据预处理：标准化
import numpy as np  # 数值计算
from sklearn.cross_validation import train_test_split  # 训练测试数据分离
from sklearn.datasets.samples_generator import make_classification  # 生成数据
from sklearn.svm import SVC  # 使用的model
import matplotlib.pyplot as plt  # 画图可视化

# # (一) 数据标准化
# a = np.array([[10, 2.7, 3.6],
#               [-100, 5, -2],
#               [120, 20, 40]], dtype=np.float64)
#
# print('原始数据', a)
# print('正规化数据', preprocessing.scale(a))

# (二) 数据标准化对机器学习的影响
X, y = make_classification(n_samples=300, n_features=2, n_redundant=0, n_informative=2,
                           random_state=22, n_clusters_per_class=1, scale=100)  # 生成数据
# X：形状数组[n_samples，n_features] 生成的样本。
# y：形状数组[n_samples] 每个样本的类成员的整数标签。
# plt.scatter(X[:, 0], X[:, 1], c=y)  # 散点图
# plt.show() # 开始显示

# 开始数据标准化
# X = preprocessing.scale(X)  # minmax_scale(X, feature_range(-1, 1))
# 使用training data进行学习，使用test data进行预测
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
Classifier = SVC()  # 选择分类器模型
Classifier.fit(X_train, y_train)  # 开始训练
print(Classifier.score(X_test, y_test))  # 打印测试得分s

