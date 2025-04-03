from loaddata import load_darpa_dataset,load_batch_level_dataset
from model import GCNModel
from torch.optim import Adam
import argparse
import torch
from tqdm import tqdm
import os
import pickle as pkl
import dgl
import torch.nn.functional as F
import torch.nn as nn
import random
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.dataloading import GraphDataLoader
import numpy as np
IN_DIM = 128
HIDDEN_DIM = 64
NUM_LAYERS = 2

def extract_dataloaders(entries, batch_size):
    random.shuffle(entries)
    train_idx = torch.arange(len(entries))
    train_sampler = SubsetRandomSampler(train_idx)
    train_loader = GraphDataLoader(entries, batch_size=batch_size, sampler=train_sampler)
    return train_loader

class Pooling(nn.Module):
    def __init__(self, pooler):
        super(Pooling, self).__init__()
        self.pooler = pooler

    def forward(self, graph, feat, t=None):
        feat = feat
        # Implement node type-specific pooling
        with graph.local_scope():
            if t is None:
                if self.pooler == 'mean':
                    return feat.mean(0, keepdim=True)
                elif self.pooler == 'sum':
                    return feat.sum(0, keepdim=True)
                elif self.pooler == 'max':
                    return feat.max(0, keepdim=True)
                else:
                    raise NotImplementedError
            elif isinstance(t, int):
                mask = (graph.ndata['type'] == t)
                if self.pooler == 'mean':
                    return feat[mask].mean(0, keepdim=True)
                elif self.pooler == 'sum':
                    return feat[mask].sum(0, keepdim=True)
                elif self.pooler == 'max':
                    return feat[mask].max(0, keepdim=True)
                else:
                    raise NotImplementedError
            else:
                mask = (graph.ndata['type'] == t[0])
                for i in range(1, len(t)):
                    mask |= (graph.ndata['type'] == t[i])
                if self.pooler == 'mean':
                    return feat[mask].mean(0, keepdim=True)
                elif self.pooler == 'sum':
                    return feat[mask].sum(0, keepdim=True)
                elif self.pooler == 'max':
                    return feat[mask].max(0, keepdim=True)
                else:
                    raise NotImplementedError


def transform_graph(g, node_feature_dim, edge_feature_dim):
    new_g = g.clone()
    new_g.ndata["attr"] = F.one_hot(g.ndata["type"].view(-1), num_classes=node_feature_dim).float()
    new_g.edata["attr"] = F.one_hot(g.edata["type"].view(-1), num_classes=edge_feature_dim).float()
    return new_g


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Darpa TC E3 Train')
    parser.add_argument("--dataset", type=str, default="wget")
    # parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    args = parser.parse_args()
    dataset = args.dataset
    lr = args.lr
    if dataset in ['cadets','theia','trace']:
        max_epoch = 50
        whole_g = load_darpa_dataset(dataset,mode='train')
        in_dim = IN_DIM
        hidden_dim = HIDDEN_DIM
        num_layers = NUM_LAYERS
        model = GCNModel(in_dim, hidden_dim, num_layers)  # build_model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.train()
        optimizer = Adam(model.parameters(), lr=lr)
        epoch_iter = tqdm(range(max_epoch))
        for epoch in epoch_iter:
            epoch_loss = 0.0
            for i in range(len(whole_g)):
                g = whole_g[i].to(device)
                model.train()
                loss = model(g)
                loss /= len(whole_g)
                optimizer.zero_grad()
                epoch_loss += loss.item()
                loss.backward(retain_graph=True)
                optimizer.step()
                del g
            epoch_iter.set_description(f"Epoch {epoch} | train_loss: {epoch_loss:.4f}")
        torch.save(model.state_dict(), "./checkpoints/checkpoint-{}.pt".format(dataset))
        save_dict_path = './eval_result/distance_save_{}.pkl'.format(dataset)
        if os.path.exists(save_dict_path):
            os.unlink(save_dict_path)
    else:
        max_epoch = 5
        hidden_dim = 256
        num_layers = 4
        batch_size = 1
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cuda:1")
        whole_data = load_batch_level_dataset()
        n_node_feat = whole_data['n_feat']
        graphs = whole_data['graph']
        train_index = whole_data['train_index']
        model = GCNModel(n_node_feat, hidden_dim, num_layers)
        model = model.to(device)
        optimizer = Adam(model.parameters(), lr=lr)
        epoch_iter = tqdm(range(max_epoch))
        # train_loader = (extract_dataloaders(train_index, batch_size))
        # for epoch in epoch_iter:
        #     model.train()
        #     loss_list = []
        #     for _, batch in enumerate(train_loader):
        #         # batch_g = [transform_graph(graphs[idx][0], n_node_feat, n_edge_feat).to(device) for idx in batch]
        #         batch_g = [graphs[idx][0] for idx in batch]
        #         batch_g = dgl.batch(batch_g).to(device) 
        #         batch_g = dgl.add_self_loop(batch_g)
        #         batch_g.to(device)
        #         model.train()
        #         loss = model(batch_g)
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()
        #         loss_list.append(loss.item())
        #         del batch_g
        #     epoch_iter.set_description(f"Epoch {epoch} | train_loss: {np.mean(loss_list):.4f}")

        # torch.save(model.state_dict(), "./checkpoints/checkpoint-wget.pt")

        for epoch in epoch_iter:
            epoch_loss = 0.0
            for i in train_index:
                g = graphs[i][0].to(device)
                g = dgl.add_self_loop(g)
                model.train()
                loss = model(g)
                loss /= len(train_index)
                optimizer.zero_grad()
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                del g
            epoch_iter.set_description(f"Epoch {epoch} | train_loss: {epoch_loss:.4f}")
        torch.save(model.state_dict(), "./checkpoints/checkpoint-{}.pt".format(dataset))
