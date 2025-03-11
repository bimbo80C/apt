from loaddata import load_darpa_dataset
from model import GCNModel, SAGENet, GMAEModel
from torch.optim import Adam
import argparse
import torch
from tqdm import tqdm


def train(model, g, lr, epochs=50):
    optimizer = Adam(model.parameters(), lr=lr)
    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()
        optimizer.zero_grad()
        loss = model(g)
        loss.backward(retain_graph=True)
        optimizer.step()

        # 打印第一个 epoch 的 loss
        if epoch == 0:
            print(f"Epoch [1/{epochs}], Initial Loss: {loss.item():.4f}")

        # 每 10 个 epoch 打印一次 loss
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")


import pickle as pkl
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Darpa TC E3 Train')
    parser.add_argument("--dataset", type=str, default="trace")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    args = parser.parse_args()
    dataset = args.dataset
    lr = args.lr
    train_g = load_darpa_dataset(dataset,mode='train')
    features = train_g.ndata['attr']
    print(f"Feature shape: {features.shape}")  # 打印节点特征的形状
    print(f"Number of nodes: {train_g.num_nodes()}")
    print(f"Number of edges: {train_g.num_edges()}")
    in_dim = features.shape[1]  # in_dim = 128
    hidden_dim = 64
    num_layers = 2

    # Ours
    # model = GCNModel(in_dim, hidden_dim, num_layers)  # build_model
    # Threatrace
    # model = SAGENet(in_dim, hidden_dim)
    # MAGIC
    model = GMAEModel(
        n_dim=in_dim,
        hidden_dim=hidden_dim,
        n_layers=num_layers,
        n_heads=4,
        activation="prelu",
        feat_drop=0.1,
        negative_slope=0.2,
        residual=True,
        mask_rate=0.5,
        norm='BatchNorm',
        loss_fn='sce',
        alpha_l=2
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_g = train_g.to(device)
    model = model.to(device)
    train(model, train_g, lr)
    torch.save(model.state_dict(), "./checkpoints/checkpoint-{}.pt".format(dataset))
from loaddata import load_darpa_dataset
from model import GCNModel, SAGENet, GMAEModel
from torch.optim import Adam
import argparse
import torch
from tqdm import tqdm
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Darpa TC E3 Train')
    parser.add_argument("--dataset", type=str, default="trace")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    args = parser.parse_args()
    dataset = args.dataset
    lr = args.lr
    # max_epoch = 16
    max_epoch = 50
    whole_g = load_darpa_dataset(dataset, mode='train')
    in_dim = 128
    hidden_dim = 64
    num_layers = 2

    # Ours
    # model = GCNModel(in_dim, hidden_dim, num_layers)  # build_model
    # Threatrace
    # model = SAGENet(in_dim, hidden_dim)
    # MAGIC
    model = GMAEModel(
        n_dim=in_dim,
        hidden_dim=hidden_dim,
        n_layers=num_layers,
        n_heads=4,
        activation="prelu",
        feat_drop=0.1,
        negative_slope=0.2,
        residual=True,
        mask_rate=0.5,
        norm='BatchNorm',
        loss_fn='sce',
        alpha_l=2
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Wrap model with DataParallel if there are multiple GPUs
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)  # This wraps the model to use multiple GPUs

    model = model.to(device)
    model.train()
    optimizer = Adam(model.parameters(), lr=lr)
    epoch_iter = tqdm(range(max_epoch))
    
    for epoch in epoch_iter:
        epoch_loss = 0.0
        for i in range(len(whole_g)):
            g = whole_g[i].to(device)
            g.edata['attr'] = torch.randn(g.num_edges(), 1).to(device)  # e_dim 是边特征的维度
            g.to(device)
            model.train()
            loss = model(g)
            loss /= len(whole_g)
            optimizer.zero_grad()
            epoch_loss += loss.item()
            loss.backward(retain_graph=True)
            optimizer.step()
            del g
            torch.cuda.empty_cache()

        epoch_iter.set_description(f"Epoch {epoch} | train_loss: {epoch_loss:.4f}")
    
    torch.save(model.state_dict(), f"./checkpoints/checkpoint-{dataset}.pt")
    save_dict_path = f'./eval_result/distance_save_{dataset}.pkl'
    if os.path.exists(save_dict_path):
        os.unlink(save_dict_path)
