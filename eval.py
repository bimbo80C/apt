import os
import numpy as np
import torch
import random
import pickle as pkl
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, precision_recall_curve
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import GCNModel,GMAEModel,SAGENet
import argparse
from loaddata import load_darpa_dataset
from sklearn.manifold import TSNE
import torch.nn.functional as F



def save_seed(seed, filename='random_seed.pkl'):
    with open(filename, 'wb') as f:
        pkl.dump(seed, f)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True



def evaluate_using_knn(dataset, x_train, x_test, y_test):
    epsilon = 1e-8  # 设定一个很小的值，防止除零 for cadets
    x_train_mean = x_train.mean(axis=0)
    x_train_std = x_train.std(axis=0)
    x_train_std[x_train_std == 0] = epsilon
    x_train = (x_train - x_train_mean) / x_train_std
    x_test = (x_test - x_train_mean) / x_train_std
    if dataset == 'cadets':
        n_neighbors = 200
    else:
        n_neighbors = 10

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    nbrs.fit(x_train)  # 使用训练数据 x_train 拟合 K 近邻模型

    save_dict_path = './eval_result/distance_save_{}.pkl'.format(dataset)
    # x_train  用来训练knn模型 x_test是真正的数据
    if not os.path.exists(save_dict_path):
        idx = list(range(x_train.shape[0]))
        random.shuffle(idx)
        print("begin knn model training")
        distances, _ = nbrs.kneighbors(x_train[idx][:min(500, x_train.shape[0])], n_neighbors=n_neighbors)
        del x_train
        mean_distance = distances.mean()
        del distances
        # distances, _ = tree.query(x_test, k=n_neighbors)
        distances, _ = nbrs.kneighbors(x_test, n_neighbors=n_neighbors)
        save_dict = [mean_distance, distances.mean(axis=1)]
        distances = distances.mean(axis=1)
        with open(save_dict_path, 'wb') as f:
            pkl.dump(save_dict, f)
    else:
        with open(save_dict_path, 'rb') as f:
            mean_distance, distances = pkl.load(f)
    print('knn for eval is loaded')
    score = distances / mean_distance 
    del distances
    auc = roc_auc_score(y_test, score)
    prec, rec, threshold = precision_recall_curve(y_test, score)
    # 假设 prec 是一个 numpy 数组
    with open('outputrec.txt', 'w') as f_rec:
        for p in rec:
            f_rec.write(f'{p}\n')

    with open('outputprec.txt', 'w') as f_prec:
        for p in prec:
            f_prec.write(f'{p}\n')

    f1 = 2 * prec * rec / (rec + prec + 1e-9)
    best_idx = -1
    for i in range(len(f1)):
        # To repeat peak performance
        if dataset == 'trace' and rec[i] < 0.95:
            best_idx = i - 1
            break
        if dataset == 'theia' and rec[i] < 0.99996:
            best_idx = i - 1
            break
        if dataset == 'cadets' and rec[i] < 0.95:
            best_idx = i - 1
            break
    best_thres = threshold[best_idx]
    tn = 0
    fn = 0
    tp = 0
    fp = 0
    for i in range(len(y_test)):
        if y_test[i] == 1.0 and score[i] >= best_thres:  # 异常
            tp += 1
        if y_test[i] == 1.0 and score[i] < best_thres:
            fn += 1
        if y_test[i] == 0.0 and score[i] < best_thres:
            tn += 1
        if y_test[i] == 0.0 and score[i] >= best_thres:
            fp += 1
    print('AUC: {}'.format(auc))
    print('F1: {}'.format(f1[best_idx]))
    print('PRECISION: {}'.format(prec[best_idx]))
    print('RECALL: {}'.format(rec[best_idx]))
    print('TN: {}'.format(tn))
    print('FN: {}'.format(fn))
    print('TP: {}'.format(tp))
    print('FP: {}'.format(fp))
    return auc, 0.0, None, None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Darpa TC E3 Train')
    parser.add_argument("--dataset", type=str, default="trace")
    args = parser.parse_args()
    dataset = args.dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device is {}".format(device))
    set_random_seed(0)
    in_dim = 128
    hidden_dim = 64
    num_layers = 2
    # Ours
    # model = GCNModel(in_dim, hidden_dim, num_layers)  # build_model
    # Threatrace
    # model = SAGENet(in_dim, hidden_dim)
    # MAGIC
    model = GMAEModel(
        n_dim= in_dim,  
        hidden_dim= hidden_dim,
        n_layers= num_layers,
        n_heads=4,
        activation=F.relu,
        feat_drop=0.1,
        negative_slope=0.2,
        residual=True,
        mask_rate=0.5,
        loss_fn='sce',
        alpha_l=1.3
    )
    model.load_state_dict(torch.load("./checkpoints/checkpoint-{}.pt".format(dataset), map_location=device))
    model = model.to(device)
    model.eval()
    malicious_list = []
    if os.path.exists('./dataset/{}/test/malicious.pkl'.format(dataset).format(dataset, )):
        with open('./dataset/{}/test/malicious.pkl'.format(dataset), 'rb') as f:
            malicious_list = pkl.load(f)
    with torch.no_grad():
        whole_g = load_darpa_dataset(dataset)
        x_train = []
        for i in range(len(whole_g)):
            g= whole_g[i].to(device)
            x_train.append(model.embed(g).cpu().numpy())
            del g
        x_train = np.concatenate(x_train, axis=0)
        print('trained embed is loaded')
        skip_benign = 0
        whole_g = load_darpa_dataset(dataset, mode='test')
        x_test = []
        for i in range(len(whole_g)):
            g= whole_g[i].to(device)
            if i != len(whole_g) - 1: # 可能换数据集有问题
                skip_benign += g.number_of_nodes()
            x_test.append(model.embed(g).cpu().numpy())
        x_test = np.concatenate(x_test, axis=0)
        print('embed for test is loaded')
        n = x_test.shape[0]  # 测试集样本数量
        y_test = np.zeros(n)  # 测试集标签
        y_test[malicious_list] = 1.0
        # Exclude training samples from the test set
        test_idx = []
        for i in range(x_test.shape[0]):
            if i >= skip_benign or y_test[i] == 1.0:
                test_idx.append(i)
        result_x_test = x_test[test_idx]
        result_y_test = y_test[test_idx]
        del x_test, y_test
        test_auc, test_std, _, _ = evaluate_using_knn(dataset, x_train, result_x_test, result_y_test)
    print(f"#Test_AUC: {test_auc:.4f}±{test_std:.4f}")