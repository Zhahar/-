import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import KFold, train_test_split, cross_val_score, cross_validate
from sklearn import svm


X_train = np.genfromtxt('./train_feature.csv', delimiter=',')
y_train = np.genfromtxt('./train_target.csv', delimiter=',')
X_test = np.genfromtxt('./test_feature.csv', delimiter=',')
model = svm.SVC(C=100, kernel='linear')
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
print(y_pred)
Support_vector = model.support_vectors_
w = model.coef_
b = model.intercept_
colors = ['red', 'green']  # 建立颜色列表
labels = ['Zero', 'One']  # 建立标签类别列表
plt.xlabel=('x1')
plt.ylabel=('x2')
y=np.array(y_pred)
# 绘图
for i in range(X_test.shape[1]):  # shape[] 类别的种类数量(2)
    plt.scatter(X_test[y == i, 0],  # 横坐标
    X_test[y== i, 1],  # 纵坐标
    c=colors[i],  # 颜色
    label=labels[i])  # 标签

if w[0, 1] != 0:
    xx = np.arange(3, 13, 0.1)
    # 最佳分类线
    yy = -w[0, 0]/w[0, 1] * xx - b/w[0, 1]
    plt.scatter(xx, yy, s=4)
    # 支持向量
    b1 = Support_vector[0, 1] + w[0, 0]/w[0, 1] * Support_vector[0, 0]
    b2 = Support_vector[1, 1] + w[0, 0]/w[0, 1] * Support_vector[1, 0]
    yy1 = -w[0, 0] / w[0, 1] * xx + b1
    plt.scatter(xx, yy1, s=4)
    yy2 = -w[0, 0] / w[0, 1] * xx + b2
    plt.scatter(xx, yy2, s=4)
else:
    xx = np.ones(100) * (-b) / w[0, 0]
    yy = np.arange(0, 10, 0.1)
    plt.scatter(xx, yy)
plt.show()
