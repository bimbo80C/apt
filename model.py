from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv, SAGEConv
from utils import sce_loss, GAT
class SAGENet(torch.nn.Module):
    def __init__(self, in_dim, out_dim, mask_rate=0.3, concat=False):
        super(SAGENet, self).__init__()
        self.mask_rate = mask_rate
        # Encoder layers
        self.encoder_conv1 = SAGEConv(in_dim, out_dim, aggregator_type='mean')
        self.encoder_conv2 = SAGEConv(out_dim, out_dim, aggregator_type='mean')
        # Decoder layers
        self.decoder_conv1 = SAGEConv(out_dim, out_dim, aggregator_type='mean')
        self.decoder_conv2 = SAGEConv(out_dim, in_dim, aggregator_type='mean')

        self.mask_token = nn.Parameter(torch.zeros(1, in_dim, dtype=torch.float32))

    def forward(self, g):
        loss = self.compute_loss(g)
        return loss

    def compute_loss(self, g):
        masked_g, (mask_nodes, _) = self.mask_nodes(g, self.mask_rate)
        h = masked_g.ndata['attr'].float()  # 确保输入特征为 float32,从掩码图中获取节点属性
        h = self.encoder_conv1(masked_g, h)
        h = F.relu(h)
        h = self.encoder_conv2(masked_g, h)
        encoded_features = h
        
        # Decoder part
        h = self.decoder_conv1(masked_g, encoded_features)
        h = F.relu(h)
        h = self.decoder_conv2(masked_g, h)
        decoded_features = h
        x_original = g.ndata["attr"][mask_nodes].float()
        x_reconstructed = decoded_features[mask_nodes]
        feature_loss = F.mse_loss(x_reconstructed,x_original)
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
    
    def embed(self, g):
        masked_g, _ = self.mask_nodes(g, self.mask_rate)
        h = masked_g.ndata['attr'].float()
        h = self.encoder_conv1(masked_g, h)
        h = F.relu(h)
        h = self.encoder_conv2(masked_g, h)
        return h

class GCNModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, mask_rate=0.3, num_heads = 4):
        super(GCNModel, self).__init__()
        self.mask_rate = mask_rate
        self.encoder = nn.ModuleList()
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
        self.decoder = nn.Linear(hidden_dim, in_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, in_dim, dtype=torch.float32))  # 设置为 torch.float32
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
    
class GMAEModel(nn.Module):
    def __init__(self, n_dim, hidden_dim, n_layers, n_heads, activation,feat_drop, negative_slope, residual, mask_rate=0.5, loss_fn="sce", alpha_l=2):
        super(GMAEModel, self).__init__()
        self._mask_rate = mask_rate
        self._output_hidden_size = hidden_dim
        self.recon_loss = nn.BCELoss(reduction='mean')
        assert hidden_dim % n_heads == 0
        enc_num_hidden = hidden_dim // n_heads # 16
        enc_nhead = n_heads # 4
        dec_in_dim = hidden_dim # 64
        dec_num_hidden = hidden_dim # 64

        # build encoder
        self.encoder = GAT(
            n_dim= n_dim, 
            hidden_dim=hidden_dim, 
            out_dim=enc_num_hidden, # 16
            n_layers= n_layers,# 2
            n_heads=enc_nhead, # 4
            n_heads_out=enc_nhead, # 4
            activation=activation,
            feat_drop=feat_drop,
            attn_drop=0.0,
            negative_slope=negative_slope,
            residual=residual,
            encoding=True,
        )

        total_hidden_dim = hidden_dim * n_layers 
        self.encoder_to_decoder = nn.Linear(total_hidden_dim, dec_in_dim) 

        self.decoder = GAT(
            n_dim=dec_in_dim,
            hidden_dim=dec_num_hidden,
            out_dim=n_dim,
            n_layers=1,
            n_heads=n_heads,
            n_heads_out=1,
            activation=activation,
            feat_drop=feat_drop,
            attn_drop=0.0,
            negative_slope=negative_slope,
            residual=residual,
            encoding=False,
        )

        self.enc_mask_token = nn.Parameter(torch.zeros(1, n_dim))
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def encoding_mask_noise(self, g, mask_rate=0.3):
        new_g = g.clone()
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=g.device)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        new_g.ndata["attr"][mask_nodes] = self.enc_mask_token

        return new_g, (mask_nodes, keep_nodes)

    def forward(self, g):
        loss = self.compute_loss(g)
        return loss

    def compute_loss(self, g):
        # Feature Reconstruction
        pre_use_g, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, self._mask_rate)
        pre_use_x = pre_use_g.ndata['attr'].to(pre_use_g.device)
        print("pre_use_x shape:", pre_use_x.shape)
        use_g = pre_use_g
        enc_rep, all_hidden = self.encoder(use_g, pre_use_x, return_hidden=True)
        enc_rep = torch.cat(all_hidden, dim=1)
        print("[DEBUG] enc_rep shape:", enc_rep.shape)
        rep = self.encoder_to_decoder(enc_rep)
        print("[DEBUG] rep shape:", enc_rep.shape)   # [num_nodes, 64]
        recon = self.decoder(pre_use_g, rep)
        print("[DEBUG] rep shape:", enc_rep.shape)   # [num_nodes, 128]
        x_init = g.ndata['attr'][mask_nodes]
        x_rec = recon[mask_nodes]
        loss = self.criterion(x_rec, x_init)
        return loss

    def embed(self, g):
        x = g.ndata['attr'].to(g.device)
        rep = self.encoder(g, x)
        return rep
