from functools import partial
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


# def sce_loss(x, y, alpha=3):
#     x = F.normalize(x, p=2, dim=-1)
#     y = F.normalize(y, p=2, dim=-1)
#     loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
#     loss = loss.mean()
#     return loss


class GCNModel(nn.Module):
    # init is ok
    # def __init__(self, in_dim, hidden_dim, num_layers, mask_rate=0.3,loss_fn="sce", alpha_l=2):
    def __init__(self, in_dim, hidden_dim, num_layers, mask_rate=0.3, num_heads = 4):
        super(GCNModel, self).__init__()
        self.mask_rate = mask_rate
        self.encoder = nn.ModuleList()
        # for i in range(num_layers):
        #     self.encoder.append(SAGEConv(
        #         in_dim if i == 0 else hidden_dim,
        #         hidden_dim,
        #         aggregator_type='mean',
        #         activation=F.relu
        #     ))
        self.encoder.append(GATConv(
            in_dim,
            hidden_dim // num_heads,  # 将hidden_dim除以num_heads以保持总特征维度
            num_heads=num_heads,
            feat_drop=0.1,  # 特征dropout
            attn_drop=0.1,  # 注意力dropout
            activation=F.relu
        ))
        for i in range(1, num_layers):
            self.encoder.append(GATConv(
                hidden_dim,
                hidden_dim // num_heads,
                num_heads=num_heads,
                feat_drop=0.1,
                attn_drop=0.1,
                activation=F.relu
            ))
        # for i in range(num_layers):
        #     self.encoder.append(GATConv(
        #         in_dim if i == 0 else hidden_dim * num_heads,
        #         hidden_dim,
        #         num_heads=num_heads,
        #         activation=F.prelu
        #     ))
            # self.encoder.append(GraphConv(in_dim if i == 0 else hidden_dim, hidden_dim, activation=F.relu))
        self.decoder = nn.Linear(hidden_dim, in_dim)
        # self.decoder = nn.Linear(hidden_dim, in_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, in_dim, dtype=torch.float32))  # 设置为 torch.float32
        # self.criterion = self.setup_loss_fn(loss_fn, alpha_l)
    def embed(self, g):
        x = g.ndata['attr'].to(g.device)
        for layer in self.encoder:
            x = layer(g, x)
            if len(x.shape) == 3:
                x = x.reshape(x.shape[0], -1)
        return x

    def forward(self, g):
        loss = self.compute_loss(g)
        return loss

    def compute_loss(self, g):
        masked_g, (mask_nodes, _) = self.mask_nodes(g, self.mask_rate)
        h = masked_g.ndata['attr'].float()  # 确保输入特征为 float32,从掩码图中获取节点属性
        # print(h.shape)
        for layer in self.encoder:

            h = layer(masked_g, h)
            if len(h.shape) == 3:
                h = h.reshape(h.shape[0], -1)
            # print(h.shape)
        recon_features = self.decoder(h)

        x_original = g.ndata['attr'][mask_nodes].float()  # 保持输入一致
        x_reconstructed = recon_features[mask_nodes]
        # DEBUG
        # print("x_reconstructed:", x_reconstructed)
        # print("x_reconstructed维度 (shape):", x_reconstructed.shape)
        # print("x_original:", x_original)
        # print("x_original维度 (shape):", x_original.shape)

        
        # feature_loss = self.criterion(x_reconstructed, x_original)
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

    # def setup_loss_fn(self, loss_fn, alpha_l):
    #     if loss_fn == "sce":
    #         criterion = partial(sce_loss, alpha=alpha_l)
    #     else:
    #         raise NotImplementedError
    #     return criterion