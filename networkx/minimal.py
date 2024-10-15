import networkx as nx
import matplotlib.pyplot as plt
import itertools

def merge_nodes_with_edges(G):
    new_G = nx.Graph()
    visited = set()  
    for node1, node2, data in G.edges(data=True):
        if node1 not in visited and node2 not in visited:
            new_node = (node1, node2)
            edge_weight = data.get('weight', 1)  

            new_G.add_node(new_node, weight=edge_weight)
            
            visited.add(node1)
            visited.add(node2)

    return new_G


G = nx.Graph()
G.add_node(1)
G.add_node(2)
G.add_edge(1,2,weight=0.5)
G.add_nodes_from([3,4,5])
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