import networkx as nx
import matplotlib.pyplot as plt
import itertools

def merge_nodes_with_edges(G):
    new_G = nx.DiGraph()
    # 原始的图当中唯一的索引是边    
    edge_to_node_map = {}
    for index,edge in enumerate(G.edges()):
        edge_to_node_map[edge] = index
        new_G.add_node(index,edge = edge)
    for u1,v1 in G.edges():
        for u2,v2 in G.edges():
            if v1 == u2:
                new_G.add_edge(edge_to_node_map[(u1,v1)],edge_to_node_map[(u2,v2)])     
    return new_G

def line_graph(G):
     L = nx.DiGraph()
     for u, v in G.edges():
         L.add_node((u,v))
     # 查找原图当中共享公共节点的边 
     for (u1,v1),(u2,v2) in itertools.combinations(G.edges(),2):
            if v1 == u2:  # 有向图中 v1 -> u2，形成连接
                L.add_edge((u1, v1), (u2, v2))
     return L
         
G = nx.DiGraph()
G.add_nodes_from([1,2,3,4,5])
G.add_edges_from([(1, 2),(2, 3),(1,4),(3,4),(3,5),(1,5)],weight=0.7)
print(G.nodes)
print(G.edges)
assert set(G.nodes) == {1,2,3,4,5}
assert sorted(G.edges) == sorted([(1, 2),(2, 3),(1,4),(3,4),(3,5),(1,5)])
print(G.number_of_nodes())
print(G.number_of_edges())
print("success")    
nx.draw(G, with_labels=True)
plt.show()

new_G = merge_nodes_with_edges(G)
print("新图中的节点:", new_G.nodes)
print("新图中的边:", new_G.edges(data=True))
nx.draw(new_G, with_labels=True)
plt.show()

new_L = line_graph(G)
print("新图中的节点:", new_L.nodes)
print("新图中的边:", new_L.edges(data=True))
nx.draw(new_L, with_labels=True)
plt.show()