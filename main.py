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
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# 读取文件并统计第5列数据

if __name__ == "__main__":
    file_path = './dataset/trace/attr_subject.txt'

    data_list = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split("\t")  # 按Tab分割
            if len(parts) >= 6:  # 确保至少有5列
                try:
                    value = int(parts[5])  # 第5列转换为整数
                    data_list.append(value)
                except ValueError:
                    pass  # 忽略无法转换的非数值行

    # 计算数据范围
    min_value = min(data_list)
    max_value = max(data_list)

    # 统计数据频率
    counter = Counter(data_list)

    # 数据种类（唯一值个数）
    unique_values = len(counter)

    # 输出结果
    print(f"第5列数据范围：{min_value} - {max_value}")
    print(f"数据种类数量：{unique_values}")
    print("前10个数据频率：")
    for value, freq in counter.most_common(10):  # 显示出现次数最多的前10个
        print(f"值 {value} 出现 {freq} 次")
    # 读取数据，假设数据是 tab 分隔的，文件名为 'data.txt'
    # df = pd.read_csv('statistics(1).txt', sep='\t', header=None)
    #
    # # 提取第三列数据
    # third_column = df[2]
    # percentiles = [i / 10000 for i in range(9990, 9999)]  # 99.90%到99.99%每隔0.01%
    # values_at_percentiles = third_column.quantile(percentiles)
    #
    # # 打印结果
    # for percentile, value in zip(percentiles, values_at_percentiles):
    #     print(f"{percentile * 100}%: {value}")

    # # 计算最大1%区间的数据
    # percentile_99 = third_column.quantile(0.9999)

    # # 提取最大1%数据
    # top_1_percent = third_column[third_column >= percentile_99]
    #
    # # 计算每个数据在最大1%数据中的占比
    # top_1_percent_count = top_1_percent.count()
    # total_count = third_column.count()
    #
    # # 绘制数据大小与占比之间的关系图
    # plt.figure(figsize=(10, 6))
    #
    # # 统计数据大小与对应占比
    # top_1_percent_sorted = top_1_percent.sort_values()
    #
    # # 计算对应的占比
    # cumulative_percentage = (top_1_percent_sorted.rank() / top_1_percent_count) * 100
    #
    # plt.plot(top_1_percent_sorted, cumulative_percentage, label="Cumulative Distribution", color='b')
    #
    # # 设置图表标签
    # plt.title("Distribution of Top 1% Data")
    # plt.xlabel("Data Value")
    # plt.ylabel("Cumulative Percentage (%)")
    # plt.grid(True)
    #
    # # 显示图例
    # plt.legend()
    # plt.show()

    # np.random.seed(42)  # 设置随机种子，确保结果可重复
    # x_train = np.random.rand(1000, 10)  # 随机生成一个 1000x10 的数组
    # idx = list(range(x_train.shape[0]))
    # # 打乱 x_train 的索引
    #
    # # 获取前 min(50000, x_train.shape[0]) 个样本的切片
    # sample_size = min(2, x_train.shape[0])  # 如果 x_train 的样本数小于 50000，则选择所有样本
    # subset = x_train[idx][:sample_size]  # 使用 idx 索引并进行切片
    #
    # # 输出结果
    # print("Subset shape:", subset.shape)
    # print("Subset data preview:\n", subset[:10])  #
    # dataset='trace'
    # g_edges_list = []
    # cnt = 0
    # if os.path.exists('./dataset/{}/entity_pair1.txt'.format(dataset)):
    #     with open('./dataset/{}/entity_pair1.txt'.format(dataset), 'r', encoding='utf-8') as f:
    #         print('processing g_edges_list')
    #         map_a = defaultdict(list)
    #         map_b = defaultdict(list)
    #         for line in f:
    #             cnt += 1
    #             event, src, dst, time = line.strip().split('\t')
    #             hash_dst = hash(dst)
    #             hash_src = hash(src)
    #             map_a[hash_dst] = cnt
    #             map_b[hash_src] = cnt
    #             if cnt >10:
    #                 break
    #         print(map_a)
    #         print(map_b)
    #         for hash_dst in tqdm(map_a, total=len(map_a)):
    #             if hash_dst in map_b:
    #                 event_src = map_a[hash_dst]
    #                 event_dst = map_b[hash_dst]
    #                 if event_src != event_dst and (event_src, event_dst) not in g_edges_list:  # 避免自环
    #                     g_edges_list.append((event_src, event_dst))
    #
    #         print(g_edges_list)
