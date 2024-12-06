import pickle as pkl
import dgl
import networkx as nx
import json
import os
import time
import torch
# g_edges_set.add((event_src, event_dst))
# g_nodes_list.append((cnt, {"attr": sub_g_embedding}))
# 将g_edge_list.pkl与g_node_list.pkl加载成dgl
def load_darpa_dataset(dataset,feature_dim=128):
    start_time = time.time()  # 记录开始时间
    g_nodes_list = []
    g_edges_list = []
    if os.path.exists('./dataset/{}/g_nodes_list.pkl'.format(dataset)):
        with open('./dataset/{}/g_nodes_list.pkl'.format(dataset), 'rb') as f:
            g_nodes_list = pkl.load(f)
    if os.path.exists('./dataset/{}/g_edges_list.pkl'.format(dataset)):
        with open('./dataset/{}/g_edges_list.pkl'.format(dataset), 'rb') as f:
            g_edges_list = pkl.load(f)
    # 剔除掉因为随机采样忽略掉的节点
    feature_map = {}
    for node in g_nodes_list:
        feature_map[node[0]]=node[1]["attr"]
    # g = dgl.DGLGraph()
    # print(len(feature_map))
    # print(len(g_edges_list))
    # g.ndata['feat'] = torch.zeros((g.number_of_nodes(), feature_dim))
    g=nx.DiGraph()
    for edge in g_edges_list:
        g.add_node(edge[0],attr=feature_map[edge[0]])
        g.add_node(edge[1],attr=feature_map[edge[1]])
        g.add_edge(edge[0], edge[1])
        # print(edge[0])
    train_g = dgl.from_networkx(g,node_attrs=['attr'])
    for i in range(3):
        print(f"节点 {i} 的属性: {train_g.ndata['attr'][i]}")
    # g = dgl.graph((src_nodes, dst_nodes))
    # g.ndata['feat'] = node_features
    # print(g)
    end_time = time.time()  # 记录结束时间
    print(f"Time taken to load and process dataset: {end_time - start_time:.4f} seconds")  # 输出运行时间
    # for i in range(g.num_nodes()):
    #     print(f"Node {i} features: {g.ndata['feat'][i]}")
    # return g
    # return g
