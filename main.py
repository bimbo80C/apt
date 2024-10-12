import os
import argparse
from tqdm import tqdm
import re
import json
import collections
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from attr_graph import GCNEncoder
import torch
import pickle

from torch_geometric.nn import global_mean_pool
if __name__ == '__main__':
#统一维度，数值型的其他维度直接用0填充
#类型的使用独热编码
#null使用-1
#有唯一性的使用编号，编号之后再进行统一维度
#word2vec
    nodes_dict = {
        0: {'uuid': '753366C8-7B00-E70F-1E95-2102227BD6E1'},
        1:{'record': 'Subject'},
        2:{'subject_type': 'SUBJECT_PROCESS'},
        3:{'parent': 'null'},
        4:{'local_principal': '29895546-B124-1BEC-E91C-C9107B81C616'},
        5:{'cid': '412'},
        6:{'start_time': '0'},
        7:{'unit_id': '0'},
        8:{'cmdline': 'null'}
    }

    for v in nodes_dict.values():
        print(list(v.values()))
        torch.tensor(list(v.values()),dtype=torch.float)
    #node_features = torch.tensor([list(v.values()) for v in nodes_dict.values()], dtype=torch.float)