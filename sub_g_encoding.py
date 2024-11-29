import torch.nn.functional as F
import dgl
import pickle as pkl
import  os
import networkx as nx
import numpy as np
import pandas as pd
import torch
def process_type(type):
    # 处理单个type属性
    if isinstance(type, (list, np.ndarray)):
        return torch.tensor(type, dtype=torch.float32)
    return torch.tensor([float(type)], dtype=torch.float32)

def convert_to_dgl(sub_g):
    # 将NetworkX图转换为DGL图，并处理混合类型的属性
    # 首先获取所有节点的type属性
    node_types = nx.get_node_attributes(sub_g, 'type')
    processed_types = {}
    max_len = 1  # 跟踪最大长度
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
    nx.set_node_attributes(sub_g, processed_types, 'type')
    return dgl.from_networkx(sub_g, node_attrs=['type'], edge_attrs=['type'])

def transform_graph(sub_g, node_feature_dim, edge_feature_dim):
    new_g = sub_g.clone()
    new_g.ndata["attr"] = F.one_hot(sub_g.ndata["type"].view(-1), num_classes=node_feature_dim).float()
    new_g.edata["attr"] = F.one_hot(sub_g.edata["type"].view(-1), num_classes=edge_feature_dim).float()
    return new_g

def sub_g_encoding(dataset):
    path = './dataset/{}'.format(dataset)
    sub_g_list = [convert_to_dgl(sub_g) for sub_g in pkl.load(open(path + '/sub_g_list.pkl', 'rb'))]
    node_feature_dim = 0
    for sub_g in sub_g_list:
        node_feature_dim = max(sub_g.ndata["type"].max().item(), node_feature_dim)
    node_feature_dim += 1
    edge_feature_dim = 0
    for sub_g in sub_g_list:
        edge_feature_dim = max(sub_g.edata["type"].max().item(), edge_feature_dim)
    edge_feature_dim += 1
    result_g = []
    for sub_g in sub_g_list:
        sub_g = transform_graph(sub_g, node_feature_dim, edge_feature_dim)
        result_g.append(g)

