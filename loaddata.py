import pickle as pkl
import dgl
import networkx as nx
import json
import os
import time
import torch
from tqdm import tqdm
import torch.nn.functional as F
INPUT_DIM = 128

class WgetDataset(dgl.data.DGLDataset):
    def process(self):
        pass
    def __init__(self, name):
        super(WgetDataset, self).__init__(name=name)
        if name == 'wget':
            path = './dataset/wget/final'
            num_graphs = 150
            self.graphs = []
            self.labels = []
            print('Loading {} dataset...'.format(name))
            for i in tqdm(range(num_graphs)):
                idx = i
                g = dgl.from_networkx(
                    nx.node_link_graph(json.load(open('{}/{}.json'.format(path, str(idx))))),
                    node_attrs=['type'],
                    edge_attrs=['type']
                )
                self.graphs.append(g)
                if 0 <= idx <= 24:
                    self.labels.append(1)
                else:
                    self.labels.append(0)
        else:
            raise NotImplementedError

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)

def transform_graph(g, node_feature_dim, edge_feature_dim):
    new_g = g.clone()
    new_g.ndata["attr"] = F.one_hot(g.ndata["type"].view(-1), num_classes=node_feature_dim).float()
    new_g.edata["attr"] = F.one_hot(g.edata["type"].view(-1), num_classes=edge_feature_dim).float()
    return new_g


def load_batch_level_dataset():
    graph_list = []
    path = './dataset/wget/entity_pair'
    num_graphs = 150
    for i in tqdm(range(num_graphs)):
        file_path = '{}/graph{}.pkl'.format(path,i+1)
        with open(file_path, "rb") as f:
            data = pkl.load(f)
            label = 1 if i < 25 else 0
            graph_list.append((data, label))
    node_feature_dim = 24
    full_dataset = [i for i in range(len(graph_list))]
    train_dataset = [i for i in range(len(graph_list)) if graph_list[i][1] == 0]

    return {'graph': graph_list,
            'train_index': train_dataset,
            'full_index': full_dataset,
            'n_feat': node_feature_dim,
            }

def load_darpa_dataset(dataset,feature_dim=INPUT_DIM,mode='train'):
    start_time = time.time()  # 记录开始时间
    g_edges_list = []
    if os.path.exists('./dataset/{}/{}/g_edges_list.pkl'.format(dataset,mode)):
        with open('./dataset/{}/{}/g_edges_list.pkl'.format(dataset,mode), 'rb') as f:
            g_edges_list = pkl.load(f)
    file_cnt=0
    for file in os.listdir('./dataset/{}/{}/'.format(dataset,mode)):
        if 'g_node' in file:
            file_cnt+=1
    whole_g = []
    print(file_cnt)
    for i in range(file_cnt):
        if os.path.exists('./dataset/{}/{}/g_nodes_list_{}.pkl'.format(dataset,mode,i)):
            with open('./dataset/{}/{}/g_nodes_list_{}.pkl'.format(dataset,mode,i), 'rb') as f:
                g_nodes = pkl.load(f)
                g = nx.DiGraph()
                g_edges = g_edges_list[i]
                for node in g_nodes:
                    g.add_node(node[0], attr=node[1]["attr"].float())
                for edge in g_edges:
                    g.add_edge(edge[0], edge[1])
                train_g = dgl.from_networkx(g, node_attrs=['attr'])
                # 获取所有节点的度
                g_degree = train_g.in_degrees() + train_g.out_degrees()
                # 统计度为 0 的节点个数
                zero_degree_count = (g_degree == 0).sum().item()
                print(f"Number of nodes: {train_g.number_of_nodes()}")
                print(f"Number of nodes with zero degree: {zero_degree_count}")
                train_g = dgl.add_self_loop(train_g)  # 添加自环？

                whole_g.append(train_g)
    end_time = time.time()  # 记录结束时间
    print(f"Time taken to load and process dataset: {end_time - start_time:.4f} seconds")  # 输出运行时间
    return whole_g
