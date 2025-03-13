import torch.nn.functional as F
import torch
import torch.nn as nn
from dgl.ops import edge_softmax
import dgl.function as fn
from dgl.utils import expand_as_pair
from functools import partial
from dgl.nn.pytorch import GATConv

class GAT(nn.Module):
    def __init__(
        self,
        n_dim,                  # 输入特征维度
        hidden_dim,             # 隐藏层总维度（需 = num_heads * per_head_dim）
        out_dim,                # 输出层每个头的维度
        n_layers,
        n_heads,                # 中间层头数
        n_heads_out,            # 输出层头数
        activation,
        feat_drop,
        attn_drop,
        negative_slope,
        residual,
        encoding=False
    ):
        super(GAT, self).__init__()
        self.n_heads = n_heads
        self.n_heads_out = n_heads_out
        self.n_layers = n_layers
        self.gats = nn.ModuleList()
        last_residual = encoding and residual

        # 第一层
        self.gats.append(GATConv(
            in_feats=n_dim,
            out_feats=hidden_dim // n_heads,  # 每个头的维度
            num_heads=n_heads,
            feat_drop=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            activation=activation
        ))

        # 中间层（n_layers >= 2 时）
        for _ in range(1, n_layers - 1):
            self.gats.append(GATConv(
                in_feats=hidden_dim,          # 输入是展平后的二维特征
                out_feats=hidden_dim // n_heads,
                num_heads=n_heads,
                feat_drop=feat_drop,
                attn_drop=attn_drop,
                negative_slope=negative_slope,
                residual=residual,
                activation=activation
            ))

        # 最后一层
        self.gats.append(GATConv(
            in_feats=hidden_dim,
            out_feats=out_dim,                
            num_heads=n_heads_out,
            feat_drop=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=last_residual,
            activation=activation
        ))

        self.head = nn.Identity()
        self.activation = activation if activation is not None else nn.Identity()

    def forward(self, g, input_feature, return_hidden=False):
        h = input_feature
        hidden_list = []
        for layer in self.gats:
            h = layer(g, h)
            # 展平多头输出： [num_nodes, num_heads, out_feats] → [num_nodes, num_heads * out_feats]
            h = h.flatten(1)
            hidden_list.append(h)
        if return_hidden:
            return self.head(h), hidden_list
        else:
            return self.head(h)

# class GAT(nn.Module):
#     def __init__(self,
#                  n_dim,
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
#                  encoding=False
#                  ):
#         super(GAT, self).__init__()
#         self.out_dim = out_dim
#         self.n_heads = n_heads
#         self.n_layers = n_layers
#         self.gats = nn.ModuleList()
#         last_residual = (encoding and residual)

#         if self.n_layers == 1:
#             self.gats.append(GATConv(
#                 in_feats =n_dim, out_feats= out_dim, num_heads= n_heads_out, feat_drop= feat_drop, attn_drop= attn_drop, negative_slope= negative_slope,
#                 residual= last_residual,activation= activation
#             ))
#         else:
#             self.gats.append(GATConv(
#                 in_feats =n_dim, out_feats= hidden_dim, num_heads= n_heads, feat_drop= feat_drop, attn_drop= attn_drop, negative_slope= negative_slope,
#                 residual= residual,activation = activation
#             ))
#             for _ in range(1, self.n_layers - 1):
#                 self.gats.append(GATConv(
#                     in_feats= hidden_dim * self.n_heads, out_feats= hidden_dim, num_heads= n_heads,
#                     feat_drop= feat_drop, attn_drop= attn_drop, negative_slope= negative_slope,
#                     residual= residual, activation = activation
#                 ))
#             self.gats.append(GATConv(
#                 in_feats= hidden_dim * self.n_heads, out_feats= out_dim, num_heads= n_heads_out,
#                 feat_drop= feat_drop, attn_drop= attn_drop, negative_slope= negative_slope,
#                 residual= last_residual,activation = activation
#             ))
#         self.head = nn.Identity()
#         self.activation = activation if activation is not None else nn.Identity()
#     def forward(self, g, input_feature, return_hidden=False):
#         h = input_feature
#         hidden_list = []
#         for layer in range(self.n_layers):
#             h = self.gats[layer](g, h)
#             hidden_list.append(h)
#         if return_hidden:
#             return self.head(h), hidden_list
#         else:
#             return self.head(h)

#     def reset_classifier(self, num_classes):
#         self.head = nn.Linear(self.num_heads * self.out_dim, num_classes)


def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()
    return loss