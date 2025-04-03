from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import SAGEConv, GATConv
# class SAGENet(torch.nn.Module):
#     def __init__(self, in_dim, out_dim, mask_rate=0.3, concat=False):
#         super(SAGENet, self).__init__()
#         self.mask_rate = mask_rate
#         # Encoder layers
#         self.encoder_conv1 = SAGEConv(in_dim, out_dim, aggregator_type='mean')
#         self.encoder_conv2 = SAGEConv(out_dim, out_dim, aggregator_type='mean')
#         # Decoder layers
#         self.decoder_conv1 = SAGEConv(out_dim, out_dim, aggregator_type='mean')
#         self.decoder_conv2 = SAGEConv(out_dim, in_dim, aggregator_type='mean')

#         self.mask_token = nn.Parameter(torch.zeros(1, in_dim, dtype=torch.float32))

#     def forward(self, g):
#         loss = self.compute_loss(g)
#         return loss

#     def compute_loss(self, g):
#         masked_g, (mask_nodes, _) = self.mask_nodes(g, self.mask_rate)
#         h = masked_g.ndata['attr'].float()  # 确保输入特征为 float32,从掩码图中获取节点属性
#         h = self.encoder_conv1(masked_g, h)
#         h = F.relu(h)
#         h = self.encoder_conv2(masked_g, h)
#         encoded_features = h
        
#         # Decoder part
#         h = self.decoder_conv1(masked_g, encoded_features)
#         h = F.relu(h)
#         h = self.decoder_conv2(masked_g, h)
#         decoded_features = h
#         x_original = g.ndata["attr"][mask_nodes].float()
#         x_reconstructed = decoded_features[mask_nodes]
#         feature_loss = F.mse_loss(x_reconstructed,x_original)
#         return feature_loss

# class GCNModel(nn.Module):
#     def __init__(self, in_dim, hidden_dim, num_layers, mask_rate=0.3, num_heads = 4):
#         super(GCNModel, self).__init__()
#         self.mask_rate = mask_rate
#         self.encoder = nn.ModuleList()
#         self.encoder.append(GATConv(
#             in_dim,
#             hidden_dim // num_heads,  # 将hidden_dim除以num_heads以保持总特征维度
#             num_heads=num_heads,
#             feat_drop=0.1,  # 特征dropout
#             attn_drop=0.1,  # 注意力dropout
#             activation=F.relu
#         ))
#         for i in range(1, num_layers):
#             self.encoder.append(GATConv(
#                 hidden_dim,
#                 hidden_dim // num_heads,
#                 num_heads=num_heads,
#                 feat_drop=0.1,
#                 attn_drop=0.1,
#                 activation=F.relu
#             ))
#         # self.decoder = nn.Linear(hidden_dim, in_dim)
#         self.decoder = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.ReLU(),
#             nn.Linear(hidden_dim // 2, in_dim)
#         )
#         self.mask_token = nn.Parameter(torch.zeros(1, in_dim, dtype=torch.float32))  # 设置为 torch.float32
#     def embed(self, g):
#         x = g.ndata['attr'].to(g.device)
#         for layer in self.encoder:
#             x = layer(g, x)
#             if len(x.shape) == 3:
#                 x = x.reshape(x.shape[0], -1)
#         return x

#     def forward(self, g):
#         loss = self.compute_loss(g)
#         return loss

#     def compute_loss(self, g):
#         masked_g, (mask_nodes, _) = self.mask_nodes(g, self.mask_rate)
#         h = masked_g.ndata['attr'].float()  # 确保输入特征为 float32,从掩码图中获取节点属性
#         # print(h.shape)
#         for layer in self.encoder:
#             h = layer(masked_g, h)
#             if len(h.shape) == 3:
#                 h = h.reshape(h.shape[0], -1)
#             # print(h.shape)
#         recon_features = self.decoder(h)

#         x_original = g.ndata['attr'][mask_nodes].float()  # 保持输入一致
#         x_reconstructed = recon_features[mask_nodes]
#         # DEBUG
#         # print("x_reconstructed:", x_reconstructed)
#         # print("x_reconstructed维度 (shape):", x_reconstructed.shape)
#         # print("x_original:", x_original)
#         # print("x_original维度 (shape):", x_original.shape)

        
#         # feature_loss = self.criterion(x_reconstructed, x_original)
#         feature_loss = F.mse_loss(x_reconstructed, x_original)
#         # print(feature_loss)
#         return feature_loss

#     def mask_nodes(self, g, mask_rate):
#         new_g = g.clone()
#         num_nodes = g.num_nodes()
#         perm = torch.randperm(num_nodes, device=g.device)
#         num_mask_nodes = int(mask_rate * num_nodes)
#         mask_nodes = perm[:num_mask_nodes]
#         keep_nodes = perm[num_mask_nodes:]

#         new_g.ndata['attr'][mask_nodes] = self.mask_token.to(new_g.device, dtype=new_g.ndata['attr'].dtype)
#         return new_g, (mask_nodes, keep_nodes)
from functools import partial
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import SAGEConv, GATConv

class GCNModel(nn.Module):
    # init is ok
    # def __init__(self, in_dim, hidden_dim, num_layers, mask_rate=0.3,loss_fn="sce", alpha_l=2):
    def __init__(self, in_dim, hidden_dim, num_layers, mask_rate=0.3, num_heads = 4):
        super(GCNModel, self).__init__()
        self.mask_rate = mask_rate
        # encoder = GCN
        self.encoder = nn.ModuleList()
        for i in range(num_layers):
            self.encoder.append(GraphConv(
                in_dim if i == 0 else hidden_dim,
                hidden_dim,
                activation=F.relu
            ))
        # encoder= GAT
        #==============
        # self.encoder = nn.ModuleList()
        # self.encoder.append(GATConv(
        #     in_dim,
        #     hidden_dim // num_heads,  # 将hidden_dim除以num_heads以保持总特征维度
        #     num_heads=num_heads,
        #     feat_drop=0.1,  # 特征dropout
        #     attn_drop=0.1,  # 注意力dropout
        #     activation=F.relu
        # ))
        # for i in range(1, num_layers):
        #     self.encoder.append(GATConv(
        #         hidden_dim,
        #         hidden_dim // num_heads,
        #         num_heads=num_heads,
        #         feat_drop=0.1,
        #         attn_drop=0.1,
        #         activation=F.relu
        #     ))
            #====================
        # decoder= linear
        # self.decoder = nn.Linear(hidden_dim, in_dim)
        # decoder= MLP
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),  # 第一层降维
            nn.ReLU(),  # 非线性激活
            nn.Linear(hidden_dim // 2, in_dim)  # 输出层恢复原始特征维度
        )
        # decoder = GCN
        # self.decoder = nn.ModuleList()
        # self.decoder.append(GraphConv(hidden_dim, hidden_dim // 2, activation=F.relu))
        # self.decoder.append(GraphConv(hidden_dim // 2, in_dim))
        # decoder = GAT
        # self.decoder = GATConv(
        #     hidden_dim,
        #     in_dim,  # 使最终输出维度匹配 128
        #     num_heads=1,
        #     feat_drop=0.1,
        #     attn_drop=0.1,
        #     activation=None  # 解码时不使用 ReLU
        # )
        self.mask_token = nn.Parameter(torch.zeros(1, in_dim, dtype=torch.float32))  # 设置为 torch.float32
        # self.criterion = self.setup_loss_fn(loss_fn, alpha_l)
    def embed(self, g):
        x = g.ndata['attr'].to(g.device)
        # GCN encoder
        for layer in self.encoder:
            x = layer(g, x)
        # GAT encoder
        # for layer in self.encoder:
        #     x = layer(g, x)
        #     if len(x.shape) == 3:
        #         x = x.reshape(x.shape[0], -1)
        return x


    def forward(self, g):
        loss = self.compute_loss(g)
        return loss

    def compute_loss(self, g):
        masked_g, (mask_nodes, _) = self.mask_nodes(g, self.mask_rate)
        h = masked_g.ndata['attr'].float()  # 确保输入特征为 float32,从掩码图中获取节点属性
        # print(h.shape)
        # embeddding for GAT encoder
        # for layer in self.encoder:
        #     h = layer(masked_g, h)
        #     if len(h.shape) == 3:
        #         h = h.reshape(h.shape[0], -1)
        # embeddding for GCN encoder
        for layer in self.encoder:
            h = layer(masked_g, h)

        # loss for decoder = MLP
        recon_features = self.decoder(h)
        # ====================
        # loss for decoder = gcn
        # for layer in self.decoder:
        #     h = layer(masked_g, h)  # 使用 masked_g 进行图卷积解码
        # recon_features = h  # 最终解码得到的特征
        #==============
        # loss for decoder = GAT
        # recon_features = self.decoder(g, h).squeeze(1)  # 这里使用完整图 g
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

    # def setup_loss_fn(self, loss_fn, alpha_l):
    #     if loss_fn == "sce":
    #         criterion = partial(sce_loss, alpha=alpha_l)
    #     else:
    #         raise NotImplementedError
    #     return criterion
    @property
    def enc_params(self):
        return self.encoder.parameters()

    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])