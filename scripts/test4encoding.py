from sklearn.preprocessing import OneHotEncoder
import numpy as np
codes = ['01A4', '01ED', '0124', '01B6', '0140', '0180','01A4']
# 将数据转换为二维数组，因为OneHotEncoder需要输入是二维的
codes_array = np.array(codes).reshape(-1, 1)

# 初始化OneHotEncoder
encoder = OneHotEncoder(sparse=False)

# 对数据进行one-hot编码
one_hot_encoded = encoder.fit_transform(codes_array)

# 输出结果
print(one_hot_encoded)
