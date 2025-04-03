import os
import numpy as np
import torch
import random
import pickle as pkl
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, precision_recall_curve
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import GCNModel
import argparse
from loaddata import load_darpa_dataset,load_batch_level_dataset,transform_graph
from sklearn.manifold import TSNE
import torch.nn.functional as F
import dgl
import torch.nn as nn
IN_DIM = 128
HIDDEN_DIM = 64
NUM_LAYERS = 2
from sklearn.decomposition import PCA
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
                mask = (graph.ndata['attr'] ==t)
                if self.pooler == 'mean': 
                    return feat[mask].mean(0, keepdim=True)
                elif self.pooler == 'sum':
                    return feat[mask].sum(0, keepdim=True)
                elif self.pooler == 'max':
                    return feat[mask].max(0, keepdim=True)
                else:
                    raise NotImplementedError
            else:
                mask = (graph.ndata['attr'] == t[0])
                for i in range(1, len(t)):
                    mask |= (graph.ndata['attr'] == t[i])
                if self.pooler == 'mean':
                    return feat[mask].mean(0, keepdim=True)
                elif self.pooler == 'sum':
                    return feat[mask].sum(0, keepdim=True)
                elif self.pooler == 'max':
                    return feat[mask].max(0, keepdim=True)
                else:
                    raise NotImplementedError
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

def evaluate_batch_level_using_knn(dataset, embeddings, labels):
    x, y = embeddings, labels
    # pca = PCA(n_components=64)
    # x = pca.fit_transform(x)
    # tsne = TSNE(n_components=2, random_state=42)
    # x_all_embedded = tsne.fit_transform(x)
    # num_samples = 125
    # x_train_embedded = x_all_embedded[:num_samples]
    # x_test_embedded = x_all_embedded[num_samples:]
    # plt.figure(figsize=(8, 6))
    # plt.scatter(x_train_embedded[:, 0], x_train_embedded[:, 1], label='Train', alpha=0.5, c='blue')
    # plt.scatter(x_test_embedded[:, 0], x_test_embedded[:, 1], label='Test', alpha=0.5, c='red')
    # plt.legend()
    # plt.title("t-SNE")
    # plt.savefig("t_sne_plot_wget.png", dpi=300, bbox_inches='tight')

    if dataset == 'streamspot':
        train_count = 400
    else:
        train_count = 100
    # n_neighbors = min(int(train_count * 0.02), 10)
    n_neighbors = 2
    benign_idx = np.where(y == 0)[0]
    attack_idx = np.where(y == 1)[0]
    set_random_seed(2025)
    np.random.shuffle(benign_idx)
    np.random.shuffle(attack_idx)
    x_train = x[benign_idx[:train_count]]
    x_test = np.concatenate([x[benign_idx[train_count:]], x[attack_idx]], axis=0)
    y_test = np.concatenate([y[benign_idx[train_count:]], y[attack_idx]], axis=0)
    x_train_mean = x_train.mean(axis=0)
    x_train_std = x_train.std(axis=0)
    epsilon = 1e-8 
    # print(f"x_train_mean: {x_train_mean}")
    # print(f"x_train_std: {x_train_std}")
    x_train_std[x_train_std == 0] = epsilon
    x_train = (x_train - x_train_mean) / x_train_std
    x_test = (x_test - x_train_mean) / x_train_std

    nbrs = NearestNeighbors(n_neighbors=n_neighbors)
    nbrs.fit(x_train)
    distances, indexes = nbrs.kneighbors(x_train, n_neighbors=n_neighbors)
    mean_distance = distances.mean() * n_neighbors / (n_neighbors - 1)
    # mean_distance = distances.mean() 
    distances, indexes = nbrs.kneighbors(x_test, n_neighbors=n_neighbors)

    score = distances.mean(axis=1) / mean_distance
    auc = roc_auc_score(y_test, score)
    prec, rec, threshold = precision_recall_curve(y_test, score)
    f1 = 2 * prec * rec / (rec + prec)
    best_idx = np.argmax(f1)
    best_thres = threshold[best_idx]

    tn = 0
    fn = 0
    tp = 0
    fp = 0
    for i in range(len(y_test)):
        if y_test[i] == 1.0 and score[i] >= best_thres:
            tp += 1
        if y_test[i] == 1.0 and score[i] < best_thres:
            fn += 1
        if y_test[i] == 0.0 and score[i] < best_thres:
            tn += 1
        if y_test[i] == 0.0 and score[i] >= best_thres:
            fp += 1
    with open('outputrec.txt', 'w') as f_rec:
        for p in rec:
            f_rec.write(f'{p}\n')

    with open('outputprec.txt', 'w') as f_prec:
        for p in prec:
            f_prec.write(f'{p}\n')
    print('AUC: {}'.format(auc))
    print('F1: {}'.format(f1[best_idx]))
    print('PRECISION: {}'.format(prec[best_idx]))
    print('RECALL: {}'.format(rec[best_idx]))
    print('TN: {}'.format(tn))
    print('FN: {}'.format(fn))
    print('TP: {}'.format(tp))
    print('FP: {}'.format(fp))
    return auc, 0.0


def evaluate_using_knn(dataset, x_train, x_test, y_test):
    epsilon = 1e-8  # 设定一个很小的值，防止除零 for cadets
    x_train_mean = x_train.mean(axis=0)
    x_train_std = x_train.std(axis=0)
    x_train_std[x_train_std == 0] = epsilon
    x_train = (x_train - x_train_mean) / x_train_std
    x_test = (x_test - x_train_mean) / x_train_std
    # 打印这些值
    file_path = 'output_values.txt'
    with open(file_path, 'w') as file:
        file.write("x_train_mean:\n")
        file.write(np.array2string(x_train_mean, precision=4, separator=',') + "\n\n")
        
        file.write("x_train_std:\n")
        file.write(np.array2string(x_train_std, precision=4, separator=',') + "\n\n")
        
        file.write("x_train:\n")
        file.write(np.array2string(x_train, precision=4, separator=',') + "\n\n")
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)
        file.write("x_test:\n")
        file.write(np.array2string(x_test, precision=4, separator=',') + "\n")

    # if dataset == 'cadets':
    #     n_neighbors = 20
    # else:
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
    output_file = 'score_output.txt'
    with open(output_file, 'w') as f:
        # 写入 score 的值
        for s in score:
            f.write(f"{s}\n")  
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
        if dataset == 'theia' and rec[i] < 0.95:
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

def batch_level_evaluation(model, pooler, device, method, dataset):
    model.eval()
    x_list = []
    y_list = []
    data = load_batch_level_dataset()
    full = data['full_index']
    graphs = data['graph']
    with torch.no_grad():
        for i in full:
            g = graphs[i][0].to(device)
            g = dgl.add_self_loop(g)
            label = graphs[i][1]
            out = model.embed(g)
            # out = pooler(g, out, [2]).cpu().numpy()
            out = out.cpu().numpy()
            y_list.append(label)
            x_list.append(out)
    x = np.concatenate(x_list, axis=0)
    y = np.array(y_list)
    if 'knn' in method:
        test_auc, test_std = evaluate_batch_level_using_knn(dataset, x, y)
    else:
        raise NotImplementedError
    return test_auc, test_std


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Darpa TC E3 Train')
    parser.add_argument("--dataset", type=str, default="wget")
    args = parser.parse_args()
    dataset = args.dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device is {}".format(device))
    set_random_seed(0)
    if dataset == 'wget':
        hidden_dim = 256
        num_layers = 4
        whole_data = load_batch_level_dataset()
        n_node_feat = whole_data['n_feat']
        model = GCNModel(n_node_feat, hidden_dim, num_layers)
        model = model.to(device)
        model.load_state_dict(torch.load("./checkpoints/checkpoint-{}.pt".format(dataset), map_location=device))
        pooler = Pooling('mean')
        test_auc, test_std = batch_level_evaluation(model, pooler, device, ['knn'], args.dataset)
    else:
        in_dim = IN_DIM
        hidden_dim = HIDDEN_DIM
        num_layers = NUM_LAYERS
        model = GCNModel(in_dim, hidden_dim, num_layers)  # build_model
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
                if i != len(whole_g) - 1: 
                    skip_benign += g.number_of_nodes()
                x_test.append(model.embed(g).cpu().numpy())
            x_test = np.concatenate(x_test, axis=0)
            print('embed for test is loaded')
            tsne = TSNE(n_components=2, random_state=42)
            # 合并训练集和测试集以确保降维时不会出现信息丢失
            num_samples = 3000  
            indices_train = np.random.choice(x_train.shape[0], size=num_samples, replace=False)
            indices_test = np.array(malicious_list)  # 假设 malicious_list 是恶意节点的索引列表
            x_train_sampled = x_train[indices_train]
            x_test_sampled = x_test[indices_test]
            # 使用 t-SNE 对训练集和测试集的特征进行降维
            x_all_sampled = np.vstack([x_train_sampled, x_test_sampled])
            x_all_embedded = tsne.fit_transform(x_all_sampled)
            # 将降维后的结果分开成训练集和测试集
            x_train_embedded = x_all_embedded[:num_samples]
            x_test_embedded = x_all_embedded[num_samples:]
            # 绘制 2D 散点图
            plt.figure(figsize=(8, 6))
            # 绘制训练集的散点图，使用不同颜色表示不同数据
            plt.scatter(x_train_embedded[:, 0], x_train_embedded[:, 1], label='Train', alpha=0.5, c='blue')
            # 绘制测试集的散点图
            plt.scatter(x_test_embedded[:, 0], x_test_embedded[:, 1], label='Test', alpha=0.5, c='red')
            # 设置图例
            plt.legend()
            # 设置标题
            plt.title("t-SNE Visualization of Train and Test Embeddings")
            # 保存图形
            plt.savefig("t_sne_plot.png", dpi=300, bbox_inches='tight')
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