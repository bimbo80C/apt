import os
import argparse
from tqdm import tqdm
import re
import json
import collections
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
# from attr_graph import GCNEncoder
import pickle

dataset='trace'
cnt=0
import dgl
import pandas as pd
import random
def get_attrs(dataset):
    input_file = f'./dataset/{dataset}/test_sub.txt'
    # 读取数据到DataFrame
    df = pd.read_csv(input_file,
                     sep='\t',
                     names=['uuid', 'record', 'subject_type', 'parent',
                            'local_principal', 'cid', 'start_time',
                            'unit_id', 'cmdline'])

    # 直接将DataFrame转换为字典，uuid作为key
    uuid_to_node_attrs = df.set_index('uuid').to_dict('index')
    print(uuid_to_node_attrs)
    input_file2 = f'./dataset/{dataset}/test_file.txt'
    # 读取数据到DataFrame
    df = pd.read_csv(input_file2,
                     sep='\t',
                     names=['uuid', 'record', 'file_type', 'epoch',
                            'permission', 'path'])
    # 直接将DataFrame转换为字典，uuid作为key
    uuid_to_node_attrs.update(df.set_index('uuid').to_dict('index'))
    print(uuid_to_node_attrs)
    return  uuid_to_node_attrs

def create_test_subgraph():
    # 创建有向图
    G = nx.DiGraph()
    # 添加节点
    G.add_node(0, type='subject')
    # 文件节点
    G.add_node(1, type='filetype')
    # 网络流节点
    G.add_node(2, type='192.168.1.100')
    # 内存节点
    G.add_node(3, type='0x7FFE0000')
    # 事件节点
    G.add_node(4, type='PROCESS_CREATE')

    G.add_node(5,type='C:\\Windows\\System32\\svchost.exe -k netsvcs')
    # 添加边
    G.add_edge(0, 1, type='WRITE')  # 进程写文件
    G.add_edge(0, 2,type='')


from sklearn.feature_extraction.text import TfidfVectorizer
import torch
if __name__ == '__main__':


    # 示例文本数据
    documents = [
        "I love programming in Python",
        "Python is a great programming language",
        "I love solving problems with Python",
        "I enjoy learning new programming languages"
    ]

    # 创建 TfidfVectorizer 对象
    vectorizer = TfidfVectorizer()

    # 使用 TfidfVectorizer 转换文本数据
    tfidf_matrix = vectorizer.fit_transform(documents)
    features = torch.Tensor(tfidf_matrix.toarray())
    print(features)
    # 输出词汇表（即所有词的列表）
    print("Vocabulary: ", vectorizer.get_feature_names_out())

    # 输出 TF-IDF 特征矩阵
    print("\nTF-IDF Matrix:")

    print(tfidf_matrix.toarray())
