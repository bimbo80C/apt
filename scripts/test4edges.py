import networkx as nx
import os
G = nx.DiGraph()
edges_set = set()
def graph_edge_construction():
    print(1)
    if os.path.exists('./dataset/test.txt'):
        with open('./dataset/test.txt', 'r', encoding='utf-8') as f:
            print('loading event_list')
            lines = f.readlines()
            for i in range(len(lines)):
                line_i = lines[i].strip().split()
                for j in range(i + 1, len(lines)):
                    line_j = lines[j].strip().split()
                    if line_i[2] == line_j[1]:
                        edges_set.add((line_i[0], line_j[0]))
        G.add_edges_from(edges_set)
    print(list(G.edges))

graph_edge_construction()            