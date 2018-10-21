# -*- coding: utf-8 -*-
# sklearn基本用法4-sklearn常用属性与功能

from sklearn import datasets
from sklearn.linear_model import LinearRegression

loaded_data = datasets.load_boston()  # 加载boston房价数据
boston_X = loaded_data.data  # 获取属性
boston_y = loaded_data.target  # 获取分类
# print(boston_X[:2, :])  # 13种属性
# print(boston_y)  # regression的目标数据

model = LinearRegression()  # 可以使用默认值
model.fit(boston_X, boston_y)  # 开始训练

# print('预测值', model.predict(boston_X[:4, :]))
# print('真实值', boston_y[:4])

# coef_和intercept_为LinearRegression的属性
print('参数系数', model.coef_)  # 如y = 0.1x + 0.3，此处输出的是x的系数0.1
# （如房屋地段可作为一个参数x1,则这里输出的是此参数的系数）
print('y轴交点', model.intercept_)  # 如y = 0.1x + 0.3，此处输出的是与y轴的交点长度0.3
# model.get_params()可以获取之前定义的参数
print('参数', model.get_params())
# model.score()则可以对model进行评分，R^2 coefficient of determination
print('评分', model.score(boston_X, boston_y))
