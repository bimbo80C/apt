from loaddata import load_darpa_dataset
from model import GCNModel
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
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

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
    train_g = load_darpa_dataset(dataset,mode='train')
    features = train_g.ndata['attr']
    in_dim = features.shape[1]  # in_dim = 128
    hidden_dim = 64
    num_layers = 2
    model = GCNModel(in_dim, hidden_dim, num_layers)  # build_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_g = train_g.to(device)
    model = model.to(device)
    train(model, train_g, lr)
    torch.save(model.state_dict(), "./checkpoints/checkpoint-{}.pt".format(dataset))
    save_dict_path = './eval_result/distance_save_{}.pkl'.format(dataset)
