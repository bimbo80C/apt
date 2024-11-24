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
if __name__ == '__main__':
    # uuid_to_node_attrs = {}
    # uuid_to_node_attrs=get_attrs(dataset)
    #
    # print("\n随机抽样检查3条记录：")
    # sample_uuids = random.sample(list(uuid_to_node_attrs.keys()), 3)
    # for uuid in sample_uuids:
    #     print(f"\nUUID: {uuid}")
    #     print(uuid_to_node_attrs[uuid])


    G = nx.Graph()

    # 方法1：使用列表
    nodes_list = [1, 2, 3, 4, 5]
    G.add_nodes_from(nodes_list)

    # 带属性的节点列表
    nodes_with_attr = [(1, {"color": "red"}),
                       (2, {"color": "blue"}),
                       (3, {"color": "green"})]
    G.add_nodes_from(nodes_with_attr)
    G.add_node(6,type='/file')
    print(G.nodes.keys())
    print(G.nodes.data())