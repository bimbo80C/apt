import os
from tqdm import tqdm
import re
import json
import collections
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
# from attr_graph import GCNEncoder
import pickle
import random
dataset='trace'
cnt=0

import pandas as pd
import random
def get_attrs(dataset):
    uuid_to_edge_attrs={}
    if os.path.exists('../dataset/{}/attr_event.txt'.format(dataset)):
        with open('../dataset/{}/attr_event.txt'.format(dataset), 'r', encoding='utf-8') as f_event:
            df = pd.read_csv(f_event,
                             sep='\t',
                             names=['uuid', 'record', 'event_type', 'seq', 'thread_id', 'src', 'dst1', 'dst2', 'size',
                                    'time'],
                             dtype={
                                 'size': 'float'
                             },
                             usecols=['uuid', 'record', 'event_type', 'seq', 'thread_id', 'size', 'time']
                             )
            uuid_to_edge_attrs.update(df.set_index('uuid').to_dict('index'))
            print(df.info)
    return uuid_to_edge_attrs
if __name__ == '__main__':
    uuid_to_edge_attrs = {}
    uuid_cnt={}
    uuid_to_edge_attrs=get_attrs(dataset)

    print("\n随机抽样检查3条记录：")
    sample_uuids = random.sample(list(uuid_to_edge_attrs.keys()), min(3, len(uuid_to_edge_attrs)))
    for uuid in sample_uuids:
        print(f"\nUUID: {uuid}")
        print(uuid_to_edge_attrs[uuid])
