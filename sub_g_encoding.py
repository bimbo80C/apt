import torch.nn.functional as F
import dgl
import pickle as pkl
import  os
import networkx as nx


def transform_graph(sub_g, node_feature_dim, edge_feature_dim):
    new_g = sub_g.clone()
    new_g.ndata["attr"] = F.one_hot(sub_g.ndata["type"].view(-1), num_classes=node_feature_dim).float()
    new_g.edata["attr"] = F.one_hot(sub_g.edata["type"].view(-1), num_classes=edge_feature_dim).float()
    return new_g

def sub_g_encoding(dataset):
    path = './dataset/{}'.format(dataset)
    sub_g_list = [dgl.from_networkx(
        nx.node_link_graph(sub_g),
        node_attrs=['type'],
        edge_attrs=['type']
    ) for sub_g in pkl.load(open(path + '/sub_g_list.pkl', 'rb'))]
    print('pkl file has loaded')
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

