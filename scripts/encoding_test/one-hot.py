import numpy as np
from sklearn.preprocessing import OneHotEncoder
def encode_data(file_path,output_file):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(line.strip())
    labels = np.arange(len(data)).reshape(-1, 1)
    encoder = OneHotEncoder(sparse=False)
    one_hot = encoder.fit_transform(labels)
file_path = '../dataset/subject_cid_catalogue.txt'
output_file = 'databin_encoded_output.txt'
encoded_data = encode_data(file_path,output_file)