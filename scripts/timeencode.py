import numpy as np
import matplotlib.pyplot as plt

def positional_encoding(timestamp, reference_time, d_model=64, max_period=10000):
    """
    参数说明：
    timestamp: 待编码的时间戳
    reference_time: 参考时间戳（基准点）
    d_model: 编码向量的维度，必须是偶数
    max_period: 最大周期，影响编码的频率范围
    """
    # 计算时间差（转换为秒）
    time_diff = (timestamp - reference_time) / 1e9
    position = np.array([time_diff])
    # 计算不同维度的频率项
    # 这里的频率呈指数递减，确保不同维度能捕捉不同尺度的时间特征
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(max_period) / d_model))

    # 创建编码向量
    pe = np.zeros(d_model)
    # 偶数位置使用正弦编码
    pe[0::2] = np.sin(position * div_term)
    # 奇数位置使用余弦编码
    pe[1::2] = np.cos(position * div_term)

    return pe
def read_timestamps_from_file(file_path):
    with open(file_path, 'r') as file:
        timestamps = [int(line.strip()) for line in file]  # 假设时间戳是整数
    return timestamps
current_time = 1.52362778847e+18  # 基准时间


reference_time = current_time
d_model = 8  # 使用较小的维度便于展示
file_path = './dataset/mini_attr_time.txt'
timestamps = read_timestamps_from_file(file_path)
encoded_time = []
for ts in timestamps:
    encoded = positional_encoding(ts, reference_time, d_model=d_model)
    encoded_time.append(encoded)