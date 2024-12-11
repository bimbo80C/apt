# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.transforms import NormalizeFeatures
# from torch_geometric.nn import GCNConv
# import torch.optim as optim
# from sklearn.metrics import accuracy_score
#
#
# dataset = #数据集
# data = dataset[0]
#
# class GCN(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(GCN, self).__init__()
#         self.conv1 = GCNConv(input_dim, hidden_dim)
#         self.conv2 = GCNConv(hidden_dim, output_dim)
#     def forward(self,data):
#         x, edge_index = data.x, data.edge_index
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = self.conv2(x, edge_index)
#         return x
#
# # 输入是节点的特征维度
# model = GCN(input_dim=dataset.num_node_features,hidden_dim=16,output_dim=dataset.num_classes)
# optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
# criterion = nn.CrossEntropyLoss()
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)
# data.to(device)
#
# def train():
#     model.train()
#     optimizer.zero_grad()
#     out = model(data)
#     loss = criterion(out[data.train_mask],data.y[data.train_mask])
#     loss.backward()
#     optimizer.step()
#     return loss.item()
# def test():
#     model.eval()
#     out = model(data)
#     pred = out.argmax(dim=1)
#     train_acc = accuracy_score(data.y[data.train_mask].cpu(),pred[data.train_mask].cpu())
#     val_acc =accuracy_score(data.y[data.val_mask].cpu(),pred[data.val_mask].cpu())
#     test_acc = accuracy_score(data.y[data.test_mask].cpu(),pred[data.test_mask].cpu())
#     return train_acc,val_acc,test_acc
# epochs = 1000
# for epoch in range(epochs):
#     loss = train()
#     train_acc,val_acc,test_acc = test()
#     if epoch % 10 == 0:
#         print(f'epoch:{epoch}, loss:{loss:.4f},train acc:{train_acc:.4f},val_acc:{val_acc:.4f},test acc:{test_acc:.4f}')
# """
# 输入： 所有的良性节点
# 输出：
#
# """