import numpy as np
from sklearn.preprocessing import OneHotEncoder

def encode_data(file_path,output_file):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(line.strip())
    num_bins = int(np.sqrt(len(data)))
    sorted_data = sorted(data)  # 确保数据有序 字符型的这句可能要删除
    bins = np.array_split(sorted_data, num_bins)

    bin_labels = np.arange(num_bins).reshape(-1, 1)
    bin_encoder = OneHotEncoder(sparse=False)
    bin_one_hot = bin_encoder.fit_transform(bin_labels)

    offsets = []
    for bin_idx, bin_data in enumerate(bins):
        offset_labels = np.arange(len(bin_data)).reshape(-1, 1)
        offset_encoder = OneHotEncoder(sparse=False)
        offset_one_hot = offset_encoder.fit_transform(offset_labels)
        for offset_idx in range(len(bin_data)):
            combined_encoding = np.concatenate((bin_one_hot[bin_idx], offset_one_hot[offset_idx]))
            offsets.append((bin_data[offset_idx], combined_encoding))
            print(combined_encoding.shape) # 16000 259维度
    with open(output_file, 'w', encoding='utf-8') as output:
        for key, value in offsets:
            output.write(f"{key}\t{','.join(map(str, value))}\n")  # 将编码值转换为字符串并写入文件
    
    return dict(offsets)

# 测试数据
file_path = '../dataset/subject_cid_catalogue.txt'
output_file = 'databin_encoded_output.txt'
encoded_data = encode_data(file_path,output_file)


