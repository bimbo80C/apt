from loaddata import load_darpa_dataset
from model import GCNModel, SAGENet, GMAEModel
from torch.optim import Adam
import argparse
import torch
from tqdm import tqdm
import os
import torch.nn.functional as F
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
        activation=F.relu,
        feat_drop=0.1,
        negative_slope=0.2,
        residual=True,
        mask_rate=0.5,
        loss_fn='sce',
        alpha_l=2
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # Wrap model with DataParallel if there are multiple GPUs
    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs!")
    #     model = torch.nn.DataParallel(model)  # This wraps the model to use multiple GPUs

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
            torch.cuda.empty_cache()

        epoch_iter.set_description(f"Epoch {epoch} | train_loss: {epoch_loss:.4f}")
    
    torch.save(model.state_dict(), f"./checkpoints/checkpoint-{dataset}.pt")
    save_dict_path = f'./eval_result/distance_save_{dataset}.pkl'
    if os.path.exists(save_dict_path):
        os.unlink(save_dict_path)
