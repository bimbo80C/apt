# 这个脚本测试networkx 将子图拼接成为一个整图的操作
import networkx as nx
import matplotlib.pyplot as plt
# 创建两个示例子图
G1 = nx.Graph()
G1.add_edges_from([(1, 2), (2, 3)])
# nx.draw(G1, with_labels=True)
G2 = nx.Graph()
G2.add_edges_from([(3, 4), (4, 5)])
# nx.draw(G2, with_labels=True)
# 合并子图
G_combined = nx.compose(G1, G2)
# 可视化合并后的图
nx.draw(G_combined, with_labels=True)
plt.show()

