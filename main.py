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

if __name__ == '__main__':
    sub_graphs = [create_test_subgraph()]

    # 保存为pickle文件
    with open('sub_g_list.pkl', 'wb') as f:
        pickle.dump(sub_graphs, f)

    # 转换为DGL格式
    dgl_graphs = [dgl.from_networkx(
        g,
        node_attrs=['type'],
        edge_attrs=['type']
    ) for g in sub_graphs]