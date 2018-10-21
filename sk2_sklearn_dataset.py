# -*- coding: utf-8 -*-
# sklearn基本用法3-强大数据库

from sklearn import datasets
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt  # 图像化工具

# loaded_data = datasets.load_boston()  # 加载boston房价数据
# boston_X = loaded_data.data  # 获取属性
# boston_y = loaded_data.target  # 获取分类
# # print(boston_X[:2, :])  # 13种属性
# # print(boston_y)  # regression的目标数据
#
# model = LinearRegression()  # 可以使用默认值
# model.fit(boston_X, boston_y)  # 开始训练
#
# print('预测值', model.predict(boston_X[:4, :]))
# print('真实值', boston_y[:4])

# 生成数据
X, y = datasets.make_regression(n_samples=100, n_features=1, n_targets=1, noise=10)  # noise=1,10,30
plt.scatter(X, y)  # 点的形式
plt.show()  # 显示图像
