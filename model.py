import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv

class GCNModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, mask_rate=0.3):
        super(GCNModel, self).__init__()
        self.mask_rate = mask_rate
        self.encoder = nn.ModuleList()
        for i in range(num_layers):
            self.encoder.append(GraphConv(in_dim if i == 0 else hidden_dim, hidden_dim, activation=F.relu))
        self.decoder = nn.Linear(hidden_dim, in_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, in_dim, dtype=torch.float32))  # 设置为 torch.float32

    def forward(self, g):
        loss = self.compute_loss(g)
        return loss

    def compute_loss(self, g):
        masked_g, (mask_nodes, _) = self.mask_nodes(g, self.mask_rate)
        h = masked_g.ndata['attr'].float()  # 确保输入特征为 float32
        for layer in self.encoder:
            h = layer(masked_g, h)
        recon_features = self.decoder(h)

        x_original = g.ndata['attr'][mask_nodes].float()  # 保持输入一致
        x_reconstructed = recon_features[mask_nodes]
        feature_loss = F.mse_loss(x_reconstructed, x_original)
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
