import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from tqdm import tqdm

class GCNEncoder(torch.nn.Module):
    """
    一个简单的 Graph Convolutional Network (GCN) 编码器模块。
    将输入的节点特征和边索引编码为图嵌入向量。
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNEncoder, self).__init__()
        # 定义两层 GCNConv 卷积层
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        """
        前向传播函数
        Args:
            x (torch.Tensor): 输入的节点特征张量，形状为 [num_nodes, in_channels]
            edge_index (torch.Tensor): 边索引张量，形状为 [2, num_edges]
        Returns:
            torch.Tensor: 编码后的图嵌入向量，形状为 [num_nodes, out_channels]
        """
        # 通过第一层 GCNConv 卷积层
        x = self.conv1(x, edge_index)
        # 应用 ReLU 激活函数
        x = F.relu(x)
        # 通过第二层 GCNConv 卷积层
        x = self.conv2(x, edge_index)
        return x
#
# def generate_graphs_in_batches(event_list, uuid_to_node_attrs, id_entity_map, batch_size=1000):
#     cnt_node = 0
#     for batch_start in tqdm(range(0, len(event_list), batch_size), desc='Processing batches'):
#         batch_event_pairs = event_list[batch_start:batch_start + batch_size]#本批次需要处理的event
#         batch_graphs = []
#         batch_embeddings = []
#         for event_pair in batch_event_pairs:
#             G = nx.Graph()
#             event_uuid = event_pair['event_uuid']
#             src_uuid = event_pair['src_uuid']
#             dst_uuid = event_pair['dst_uuid']
#
#             # 构建节点和边字典
#             nodes_dict = {}
#             edges_dict = {}
#
#             src_record = id_entity_map[src_uuid]
#             dst_record = id_entity_map[dst_uuid]
#
#             if src_record == 'Subject':
#                 cnt_src_num = cnt_node
#                 node_attrs = uuid_to_node_attrs[src_uuid]
#                 for attr_name, attr_value in node_attrs.items():
#                     nodes_dict[cnt_node] = {attr_name: attr_value}
#                     cnt_node += 1
#                 for i in range(cnt_src_num + 1, cnt_node):
#                     edges_dict[(cnt_src_num, i)] = None
#
#             G.add_nodes_from(nodes_dict.items())
#             G.add_edges_from(edges_dict.keys())
#
#             # 编码和池化聚合
#             node_features = torch.tensor([list(v.values()) for v in nodes_dict.values()], dtype=torch.float)
#             edge_index = torch.tensor([[k[0], k[1]] for k in edges_dict.keys()], dtype=torch.long).t()
#
#             encoder = GCNEncoder(node_features.shape[1], 32, 64)
#             graph_embedding = global_mean_pool(encoder(node_features, edge_index), torch.tensor([0] * len(nodes_dict)))
#
#             batch_graphs.append(G)
#             batch_embeddings.append(graph_embedding)
#         yield batch_graphs, batch_embeddings
#
#
# def attr_graph_construction(dataset):
#     event_list = dataset['event_list']
#     uuid_to_node_attrs = dataset['uuid_to_node_attrs']
#     id_entity_map = dataset['id_entity_map']
#
#     graph_generator = generate_graphs_in_batches(event_list, uuid_to_node_attrs, id_entity_map)
#     graph_list = []
#     graph_embedding_list = []
#     for batch_graphs, batch_embeddings in tqdm(graph_generator, total=len(event_list) // 1000,
#                                                desc='Collecting graphs'):
#         graph_list.extend(batch_graphs)
#         graph_embedding_list.extend(batch_embeddings)
#
#     return graph_list, graph_embedding_list
#
#
# # 使用示例
# dataset = {
#     'event_list': event_list,
#     'uuid_to_node_attrs': uuid_to_node_attrs,
#     'id_entity_map': id_entity_map
# }
# graph_list, graph_embedding_list = attr_graph_construction(dataset)