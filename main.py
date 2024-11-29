# # import os
# # import argparse
# # from tqdm import tqdm
# # import re
# # import json
# # import collections
# # from collections import Counter
# # import matplotlib.pyplot as plt
# # import networkx as nx
# # import numpy as np
# # # from attr_graph import GCNEncoder
# # import pickle
# #
# # dataset='trace'
# # cnt=0
# # import dgl
# # import pandas as pd
# # import random
# # def get_attrs(dataset):
# #     input_file = f'./dataset/{dataset}/test_sub.txt'
# #     # 读取数据到DataFrame
# #     df = pd.read_csv(input_file,
# #                      sep='\t',
# #                      names=['uuid', 'record', 'subject_type', 'parent',
# #                             'local_principal', 'cid', 'start_time',
# #                             'unit_id', 'cmdline'])
# #
# #     # 直接将DataFrame转换为字典，uuid作为key
# #     uuid_to_node_attrs = df.set_index('uuid').to_dict('index')
# #     print(uuid_to_node_attrs)
# #     input_file2 = f'./dataset/{dataset}/test_file.txt'
# #     # 读取数据到DataFrame
# #     df = pd.read_csv(input_file2,
# #                      sep='\t',
# #                      names=['uuid', 'record', 'file_type', 'epoch',
# #                             'permission', 'path'])
# #     # 直接将DataFrame转换为字典，uuid作为key
# #     uuid_to_node_attrs.update(df.set_index('uuid').to_dict('index'))
# #     print(uuid_to_node_attrs)
# #     return  uuid_to_node_attrs
# #
# # def create_test_subgraph():
# #     # 创建有向图
# #     G = nx.DiGraph()
# #     # 添加节点
# #     G.add_node(0, type='subject')
# #     # 文件节点
# #     G.add_node(1, type='filetype')
# #     # 网络流节点
# #     G.add_node(2, type='192.168.1.100')
# #     # 内存节点
# #     G.add_node(3, type='0x7FFE0000')
# #     # 事件节点
# #     G.add_node(4, type='PROCESS_CREATE')
# #
# #     G.add_node(5,type='C:\\Windows\\System32\\svchost.exe -k netsvcs')
# #     # 添加边
# #     G.add_edge(0, 1, type='WRITE')  # 进程写文件
# #     G.add_edge(0, 2,type='')
# #
# #
# # from sklearn.feature_extraction.text import TfidfVectorizer
# # import torch
# # if __name__ == '__main__':
# #     #
# #     #
# #     # # 示例文本数据
# #     # documents = [
# #     #     "I love programming in Python",
# #     #     "Python is a great programming language",
# #     #     "I love solving problems with Python",
# #     #     "I enjoy learning new programming languages"
# #     # ]
# #     #
# #     # # 创建 TfidfVectorizer 对象
# #     # vectorizer = TfidfVectorizer()
# #     #
# #     # # 使用 TfidfVectorizer 转换文本数据
# #     # tfidf_matrix = vectorizer.fit_transform(documents)
# #     # features = torch.Tensor(tfidf_matrix.toarray())
# #     # print(features)
# #     # # 输出词汇表（即所有词的列表）
# #     # print("Vocabulary: ", vectorizer.get_feature_names_out())
# #     #
# #     # # 输出 TF-IDF 特征矩阵
# #     # print("\nTF-IDF Matrix:")
# #     #
# #     # print(tfidf_matrix.toarray())
# #     # 示例数据
# #     data = {'type': ['cat', 'dog', 'dog', 'bird', 'cat', 'bird', 'fish']}
# #     df = pd.DataFrame(data)
# #
# #     # 使用 Categorical 转换为类别型数据
# #     df['type'] = pd.Categorical(df['type']).codes
# #     print(df['type'])
# import networkx as nx
# import dgl
# import torch
# import pickle as pkl
#
# def ensure_tensor(value):
#     """确保属性值是 Tensor 类型"""
#     if isinstance(value, torch.Tensor):
#         return value
#     else:
#         return torch.tensor([value], dtype=torch.float32)
# # 创建示例数据
# def create_demo_data():
#     # 创建一个简单的NetworkX图
#     g1 = nx.DiGraph()
#     # 添加节点和节点属性
#     g1.add_node(0, type=torch.tensor([1.5, 0, 0, 0]))  # 直接使用tensor
#     g1.add_node(1, type=torch.tensor([0, 1, 0, 0.5]))  # 直接使用tensor
#     g1.add_node(2, type=ensure_tensor(2.0))
#     # 添加边和边属性
#     g1.add_edge(0, 1, type=ensure_tensor(1))
#     g1.add_edge(1, 2, type=ensure_tensor(2))
#
#     g2 = nx.DiGraph()
#     g2.add_node(0, type=torch.tensor([1, 0, 0, 0]))  # 直接使用tensor
#     g2.add_node(1, type=torch.tensor([2.0, 0, 0, 0]))  # 直接使用tensor
#     g2.add_edge(0, 1, type=ensure_tensor(1))
#
#     # 将图转换为node_link格式并保存
#     sub_g_list = [nx.node_link_data(g1), nx.node_link_data(g2)]
#
#     return sub_g_list
#
#
# # 示例数据的使用
# def demo():
#     # 创建示例数据
#     sub_g_list_original = create_demo_data()
#
#     # 打印原始NetworkX图的格式
#     print("原始node_link格式的图数据:")
#     print(sub_g_list_original[0])
#     print("\n")
#
#     # 转换为DGL图
#     sub_g_list = [dgl.from_networkx(
#         nx.node_link_graph(sub_g),
#         node_attrs=['type'],
#         edge_attrs=['type']
#     ) for sub_g in sub_g_list_original]
#
#     # 打印转换后的DGL图信息
#     print("转换后的DGL图信息:")
#     print(f"图的数量: {len(sub_g_list)}")
#     for i, g in enumerate(sub_g_list):
#         print(f"\n图 {i}:")
#         print(f"节点数: {g.number_of_nodes()}")
#         print(f"边数: {g.number_of_edges()}")
#         print(f"节点属性: {g.ndata['type']}")
#         print(f"边属性: {g.edata['type']}")
#
#
# if __name__ == "__main__":
#     demo()
import networkx as nx
import dgl
import torch
import numpy as np


def process_type_attr(attr):
    """处理单个type属性"""
    if isinstance(attr, (list, np.ndarray)):
        return torch.tensor(attr, dtype=torch.float32)
    return torch.tensor([float(attr)], dtype=torch.float32)


# 创建示例数据
def create_demo_data():
    # 创建一个简单的NetworkX图
    g1 = nx.DiGraph()
    # 添加节点和节点属性
    g1.add_node(0, type=1.5)
    g1.add_node(1, type=[0, 1, 0, 0.5])
    g1.add_node(2, type=2.0)
    # 添加边和边属性
    g1.add_edge(0, 1, type=1)
    g1.add_edge(1, 2, type=2)

    g2 = nx.DiGraph()
    g2.add_node(0, type=[1, 0, 0, 0])
    g2.add_node(1, type=3.0)
    g2.add_edge(0, 1, type=1)

    return [g1, g2]  # 直接返回NetworkX图对象


def convert_to_dgl(nx_graph):
    """将NetworkX图转换为DGL图，并处理混合类型的属性"""
    # 首先获取所有节点的type属性
    node_types = nx.get_node_attributes(nx_graph, 'type')

    # 预处理所有节点的type属性
    processed_types = {}
    max_len = 1  # 跟踪最大长度

    # 第一次遍历确定最大长度
    for node, attr in node_types.items():
        if isinstance(attr, (list, np.ndarray)):
            max_len = max(max_len, len(attr))

    # 第二次遍历处理所有属性
    for node, attr in node_types.items():
        if isinstance(attr, (list, np.ndarray)):
            processed_types[node] = attr
        else:
            # 将单个数值扩展为与向量相同的维度
            processed_types[node] = [float(attr)] + [0.0] * (max_len - 1)

    # 更新图的节点属性
    nx.set_node_attributes(nx_graph, processed_types, 'type')

    # 转换为DGL图
    return dgl.from_networkx(nx_graph, node_attrs=['type'], edge_attrs=['type'])


def demo():
    # 创建示例数据
    nx_graphs = create_demo_data()

    # 转换为DGL图
    sub_g_list = [convert_to_dgl(g) for g in nx_graphs]

    # 打印转换后的DGL图信息
    print("转换后的DGL图信息:")
    print(f"图的数量: {len(sub_g_list)}")
    for i, g in enumerate(sub_g_list):
        print(f"\n图 {i}:")
        print(f"节点数: {g.number_of_nodes()}")
        print(f"边数: {g.number_of_edges()}")
        print(f"节点属性: {g.ndata['type']}")
        print(f"边属性: {g.edata['type']}")

import pandas as pd
import ipaddress
import os
cnt = 0
def ip_to_binary_list(ip,cnt):
    # print(ip)
    if pd.isna(ip):
        print('{} ip is none'.format(ip))
        cnt+=1
        return None
    ip_int = int(ipaddress.ip_address(ip))
    binary_list = [(ip_int >> (127 - i)) & 1 for i in range(128)]
    binary_tensor = torch.tensor(binary_list, dtype=torch.int32)
    return binary_tensor

def get_cnt(df, attr_type):
    if attr_type in ['remote_address', 'memory_address', 'local_address']:
        df[attr_type] = df[attr_type].apply(ip_to_binary_list)
        if attr_type == 'remote_address':
            print(df[attr_type])
        return df
    else:
        return df
from collections import defaultdict, Counter
from tqdm import tqdm
if __name__ == "__main__":
    dataset='trace'
    g_edges_list = []
    cnt = 0
    if os.path.exists('./dataset/{}/entity_pair1.txt'.format(dataset)):
        with open('./dataset/{}/entity_pair1.txt'.format(dataset), 'r', encoding='utf-8') as f:
            print('processing g_edges_list')
            map_a = defaultdict(list)
            map_b = defaultdict(list)
            for line in f:
                cnt += 1
                event, src, dst, time = line.strip().split('\t')
                hash_dst = hash(dst)
                hash_src = hash(src)
                map_a[hash_dst] = cnt
                map_b[hash_src] = cnt
                if cnt >10:
                    break
            print(map_a)
            print(map_b)
            for hash_dst in tqdm(map_a, total=len(map_a)):
                if hash_dst in map_b:
                    event_src = map_a[hash_dst]
                    event_dst = map_b[hash_dst]
                    if event_src != event_dst and (event_src, event_dst) not in g_edges_list:  # 避免自环
                        g_edges_list.append((event_src, event_dst))

            print(g_edges_list)
