import torch
src_val_embed = torch.randn(1, 128)  # 源节点嵌入向量
dst_val_embed = torch.randn(1, 128)  # 目标节点嵌入向量 
edgetype_embed = torch.randn(1, 128)  # 边类型嵌入向量
node_embeddings = [src_val_embed, dst_val_embed, edgetype_embed]
    
    # 将所有嵌入向量堆叠后求和
    # 将所有嵌入向量拼接在一起
sub_g_embedding = torch.cat(node_embeddings, dim=1)
print(sub_g_embedding.shape)  # 输出嵌入向量的形状