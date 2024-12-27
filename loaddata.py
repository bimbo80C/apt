import pickle as pkl
import dgl
import networkx as nx
import json
import os
import time
import torch

def load_darpa_dataset(dataset,feature_dim=128,mode='train'):
    start_time = time.time()  # 记录开始时间
    g_nodes_list = []
    g_edges_list = []
    if os.path.exists('./dataset/{}/{}/g_edges_list.pkl'.format(dataset,mode)):
        with open('./dataset/{}/{}/g_edges_list.pkl'.format(dataset,mode), 'rb') as f:
            g_edges_list = pkl.load(f)
    if os.path.exists('./dataset/{}/{}/g_nodes_list.pkl'.format(dataset,mode)):
        with open('./dataset/{}/{}/g_nodes_list.pkl'.format(dataset,mode), 'rb') as f:
            g_nodes_list = pkl.load(f)
    g = nx.DiGraph()
    #Test whether all malicious nodes are covered
    # malicious_list = []
    # if_in_node=[]
    # node_set=set()
    # if os.path.exists('./dataset/{}/test/malicious.pkl'.format(dataset).format(dataset)):
    #     with open('./dataset/{}/test/malicious.pkl'.format(dataset), 'rb') as f:
    #         malicious_list = pkl.load(f)
    # for node in g_nodes_list:
    #     node_set.add(node[0])
    # for malicious in malicious_list:
    #     if malicious not in node_set:
    #         if_in_node.append(malicious)
    # print(if_in_node)
    #===============================
    for node in g_nodes_list:
        g.add_node(node[0],attr=node[1]["attr"].float())
    for edge in g_edges_list:
        g.add_edge(edge[0], edge[1])
    train_g = dgl.from_networkx(g,node_attrs=['attr'])
    # 获取所有节点的入度
    in_degrees = train_g.in_degrees()  # 返回一个包含所有节点入度的张量
    # 统计入度为 0 的节点个数
    zero_in_degree_count = (in_degrees == 0).sum().item()
    print(f"Number of nodes with zero in-degree: {zero_in_degree_count}")
    train_g = dgl.add_self_loop(train_g) # 添加自环
    # for i in range(3):
    #     print(f"节点 {i} 的属性: {train_g.ndata['attr'][i]}")
    end_time = time.time()  # 记录结束时间
    print(f"Time taken to load and process dataset: {end_time - start_time:.4f} seconds")  # 输出运行时间
    return train_g
