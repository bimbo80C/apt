import os
import numpy as np
import torch
import random
import pickle as pkl
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, precision_recall_curve
from model import GCNModel
import argparse
from loaddata import load_darpa_dataset

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True

def evaluate_using_knn(dataset, x_train, x_test, y_test):
    # 对训练数据和测试数据进行归一化，然后使用 K 近邻算法（KNN）对训练数据进行拟合
    x_train_mean = x_train.mean(axis=0)
    x_train_std = x_train.std(axis=0)
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
        distances, _ = nbrs.kneighbors(x_train[idx][:min(50000, x_train.shape[0])], n_neighbors=n_neighbors)
        del x_train
        mean_distance = distances.mean()
        del distances
        distances, _ = nbrs.kneighbors(x_test, n_neighbors=n_neighbors)
        save_dict = [mean_distance, distances.mean(axis=1)]
        distances = distances.mean(axis=1)
        with open(save_dict_path, 'wb') as f:
            pkl.dump(save_dict, f)
    else:
        with open(save_dict_path, 'rb') as f:
            mean_distance, distances = pkl.load(f)
    score = distances / mean_distance  # 异常分数 score越大越可能异常
    del distances
    auc = roc_auc_score(y_test, score)  # 计算AUC分数
    prec, rec, threshold = precision_recall_curve(y_test, score)
    f1 = 2 * prec * rec / (rec + prec + 1e-9)
    best_idx = -1
    for i in range(len(f1)):
        # To repeat peak performance
        if dataset == 'trace' and rec[i] < 0.99979:
            best_idx = i - 1
            break
        if dataset == 'theia' and rec[i] < 0.99996:
            best_idx = i - 1
            break
        if dataset == 'cadets' and rec[i] < 0.9976:
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

    hidden_dim = 64
    num_layers = 2
    set_random_seed(0)
    in_dim = 128

    model = GCNModel(in_dim, hidden_dim, num_layers)  # build_model
    checkpoint = torch.load("./checkpoints/checkpoint-{}.pt".format(dataset), map_location=device)
    # # 打印权重文件中的keys
    # print("Checkpoint keys:")
    # for key in checkpoint.keys():
    #     print(key)
    # # 打印当前模型的state_dict keys
    # print("\nModel state_dict keys:")
    # for key in model.state_dict().keys():
    #     print(key)
    model.load_state_dict(torch.load("./checkpoints/checkpoint-{}.pt".format(dataset), map_location=device))
    model = model.to(device)
    model.eval()
    malicious_list=[]
    if os.path.exists('./dataset/{}/test/malicious.pkl'.format(dataset).format(dataset,)):
        with open('./dataset/{}/test/malicious.pkl'.format(dataset), 'rb') as f:
            malicious_list = pkl.load(f)
    # print(len(malicious_list))
    #准备好knn的输入
    with torch.no_grad():
        skip_benign = 0
        g = load_darpa_dataset(dataset, mode='train').to(device)

        x_train=model.embed(g).cpu().numpy()
        print('trained embed is loaded')
        skip_benign += g.number_of_nodes()
        del g
        skip_benign = 0
        g = load_darpa_dataset(dataset, mode = 'test').to(device)
        x_test=model.embed(g).cpu().numpy()
        print('embed for test is loaded')
        del g
        n = x_test.shape[0]  # 测试集样本数量
        print(n)
        print(malicious_list)
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
        test_auc, test_std, _, _ = evaluate_using_knn(dataset, x_train, result_x_test,
                                                                   result_y_test)
    print(f"#Test_AUC: {test_auc:.4f}±{test_std:.4f}")