import numpy as np
from sklearn import preprocessing

enc = preprocessing.OneHotEncoder()  # 创建对象
array1 = np.asarray([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
enc.fit(array1)  # 拟合
array2 = enc.transform([[0, 1, 3]]).toarray()  # 转化
print("4个样本的特征向量是：\n", array1, '\n')
print('对第5个样本的特征向量进行one-hot：', array2)
