import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import SAGEConv, GATConv
# class GCN(nn.Module):
#     def __init__(self,
#                  n_dim,
#                  e_dim,
#                  hidden_dim,
#                  out_dim,
#                  n_layers,
#                  n_heads,
#                  n_heads_out,
#                  activation,
#                  feat_drop,
#                  attn_drop,
#                  negative_slope,
#                  residual,
#                  norm,
#                  concat_out=False,
#                  encoding=False):
class GCNConv(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers=2, activation=nn.ReLU(), dropout=0.0):
        super(GCNConv, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        # First layer
        self.layers.append(GCNConv(in_dim, hidden_dim))

        # Hidden layers
        for _ in range(n_layers - 2):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))

        # Output layer
        self.layers.append(GCNConv(hidden_dim, out_dim))

    def forward(self, x, edge_index):
        for conv in self.layers[:-1]:
            x = conv(x, edge_index)
            x = self.activation(x)
            x = self.dropout(x)

        # Final layer without activation for output
        x = self.layers[-1](x, edge_index)
        return x

class GCNModel(nn.Module):
    # init is ok
    def __init__(self, in_dim, hidden_dim, num_layers, mask_rate=0.3):
        super(GCNModel, self).__init__()
        self.mask_rate = mask_rate
        self.encoder = nn.ModuleList()
        for i in range(num_layers):
            self.encoder.append(GraphConv(in_dim if i == 0 else hidden_dim, hidden_dim, activation=F.relu))
        self.decoder = nn.Linear(hidden_dim, in_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, in_dim, dtype=torch.float32))  # 设置为 torch.float32

    def embed(self, g):
        x = g.ndata['attr'].to(g.device)
        for layer in self.encoder:
            x = layer(g, x)
        return x

    def forward(self, g):
        loss = self.compute_loss(g)
        return loss

    def compute_loss(self, g):
        masked_g, (mask_nodes, _) = self.mask_nodes(g, self.mask_rate)
        h = masked_g.ndata['attr'].float()  # 确保输入特征为 float32,从掩码图中获取节点属性
        for layer in self.encoder:
            h = layer(masked_g, h)
        recon_features = self.decoder(h)

        x_original = g.ndata['attr'][mask_nodes].float()  # 保持输入一致
        x_reconstructed = recon_features[mask_nodes]
        feature_loss = F.mse_loss(x_reconstructed, x_original)
        # print(feature_loss)
        return feature_loss

    def mask_nodes(self, g, mask_rate):
        new_g = g.clone()
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=g.device)
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[:num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        new_g.ndata['attr'][mask_nodes] = self.mask_token.to(new_g.device, dtype=new_g.ndata['attr'].dtype)
        return new_g, (mask_nodes, keep_nodes)
