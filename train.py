from loaddata import load_darpa_dataset
from model import GCNModel
from torch.optim import Adam
import argparse
import torch
from tqdm import tqdm
import os


import pickle as pkl
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Darpa TC E3 Train')
    parser.add_argument("--dataset", type=str, default="trace")
    parser.add_argument("--mode", type=str, default="trains")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    args = parser.parse_args()
    dataset = args.dataset
    lr = args.lr
    max_epoch = 50

    whole_g = load_darpa_dataset(dataset,mode='test')
    # features = train_g.ndata['attr']
    # in_dim = features.shape[1]  # in_dim = 128
    in_dim = 128
    hidden_dim = 64
    num_layers = 2
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
    os.unlink(save_dict_path)
