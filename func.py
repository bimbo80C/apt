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
# from attr_graph import GCNEncoder
# import torch
# from torch_geometric.nn import global_mean_pool

# metadata = {
#     'trace': {
#         'train': ['ta1-trace-e3-official-1.json.0', 'ta1-trace-e3-official-1.json.1', 'ta1-trace-e3-official-1.json.2',
#                   'ta1-trace-e3-official-1.json.3'],
#         'test': ['ta1-trace-e3-official-1.json.0', 'ta1-trace-e3-official-1.json.1', 'ta1-trace-e3-official-1.json.2',
#                  'ta1-trace-e3-official-1.json.3', 'ta1-trace-e3-official-1.json.4']
#     },
#     'theia': {
#         'train': ['ta1-theia-e3-official-6r.json', 'ta1-theia-e3-official-6r.json.1', 'ta1-theia-e3-official-6r.json.2',
#                   'ta1-theia-e3-official-6r.json.3'],
#         'test': ['ta1-theia-e3-official-6r.json.8']
#     },
#     'cadets': {
#         'train': ['ta1-cadets-e3-official.json', 'ta1-cadets-e3-official.json.1', 'ta1-cadets-e3-official.json.2',
#                   'ta1-cadets-e3-official-2.json.1'],
#         'test': ['ta1-cadets-e3-official-2.json']
#     }
# }

###### for all
pattern_uuid = re.compile(r'uuid\":\"(.*?)\"')
pattern_record = re.compile(r'datum\":\{\"com.bbn.tc.schema.avro.cdm18.(.*?)\"')
pattern_type = re.compile(r'type\":\"(.*?)\"')
pattern_epoch = re.compile(r'epoch\":\{\"int\":(.*?)\}')
pattern_pid = re.compile(r'map\":\{\"pid\":\"(.*?)\"\}')
pattern_size = re.compile(r'size\":\{\"long\":(.*?)\}')
###### for SrcSinkObject(uuid,epoch,pid)
pattern_fileDescriptor = re.compile(r'fileDescriptor\":\{\"int\":(.*?)\}')
###### for Principal(uuid,type)
pattern_user_id = re.compile(r'userId\":\"(.*?)\"')
pattern_group_ids = re.compile(r'groupIds\":\[\"(.*?)\",\"(.*?)\"\],')
pattern_euid = re.compile(r'euid\":\"(.*?)\"')
###### for Subject(uuid,type)
pattern_cid = re.compile(r'cid\":(.*?),')
pattern_parent = re.compile(r'parentSubject\":\{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
pattern_local_principal = re.compile(r'localPrincipal\":\"(.*?)\",')
pattern_start_time = re.compile(r'startTimestampNanos\":(.*?),')
pattern_unit_id = re.compile(r'unitId\":\{\"int\":(.*?)\},')
pattern_cmdline = re.compile(r'cmdLine\":\{\"string\":\"(.*?)\"\}')
pattern_properties1 = re.compile(
    r'properties\":\{\"map\":\{\"map\":\{\"name\":\"(.*?)\",\"seentime\":\"(.*?)\",\"ppid\":\"(.*?)\"')
pattern_properties2 = re.compile(
    r'properties\":\{\"map\":\{\"map\":\{\"name\":\"(.*?)\",\"cwd\":\"(.*?)\",\"ppid\":\"(.*?)\"')
###### for FileObject(uuid,type,epoch)
pattern_permission = re.compile(r'com.bbn.tc.schema.avro.cdm18.SHORT\":\"(.*?)\"')
pattern_path = re.compile(r'path\":\"(.*?)\"')
###### for NetFlowObject(uuid,epoch)
pattern_local_address = re.compile(r'localAddress\":\"(.*?)\"')
pattern_local_port = re.compile(r'localPort\":(.*?),')
pattern_remote_address = re.compile(r'remoteAddress\":\"(.*?)\"')
pattern_remote_port = re.compile(r'remotePort\":(.*?),')
pattern_ip_protocol = re.compile(r'ipProtocol\":\{\"int\":(.*?)\},')
###### for MemoryObject(uuid,epoch,size)
pattern_memory_address = re.compile(r'memoryAddress\":(.*?),')
pattern_tgid = re.compile(r'tgid\":\"(.*?)\"')
###### for UnnamedPipeObject(uuid,epoch,pid)
pattern_sourceFileDescriptor = re.compile(r'sourceFileDescriptor\":\{\"int\":(.*?)\}')
pattern_sinkFileDescriptor = re.compile(r'sinkFileDescriptor\":\{\"int\":(.*?)\}')
###### for event(uuid,type,size)
pattern_seq = re.compile(r'sequence\":\{\"long\":(.*?)\}')
pattern_thread_id = re.compile(r'threadId\":\{\"int\":(.*?)\}')
pattern_src = re.compile(r'subject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
pattern_dst1 = re.compile(r'predicateObject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
pattern_dst2 = re.compile(r'predicateObject2\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
pattern_time = re.compile(r'timestampNanos\":(.*?),')
name_map = {
    "Subject": "subject",
    "SrcSinkObject": "src",
    "Principal": "principal",
    "Event": "event",
    "FileObject": "file",
    "NetFlowObject": "netflow",
    "MemoryObject": "memory",
    "UnnamedPipeObject": "unnamed"
}
# TODO 将全部的节点属性进行一个if判断作为图嵌入向量部分的输入
###############
# 提取每个属性到TXT，生成id_entity_map.json即uuid和类型的映射
###############
def preprocess(dataset):
    id_entity_map = {}
    for file in os.listdir('./dataset/{}/'.format(dataset)):  #file ta1-trace-e3-official-1.json except 5 & 6 for testing
        if 'json' in file and 'entity' not in file and '5' not in file and '6' not in file: 
            print('reading {} ...'.format(file))
            f = open('./dataset/{}/'.format(dataset) + file, 'r', encoding='utf-8')
            fw_src = open('./dataset/{}/'.format(dataset) + 'attr_src.txt', 'a', encoding='utf-8') 
            fw_principal = open('./dataset/{}/'.format(dataset) + 'attr_principal.txt', 'a', encoding='utf-8')
            fw_subject = open('./dataset/{}/'.format(dataset) + 'attr_subject.txt', 'a', encoding='utf-8')
            fw_file = open('./dataset/{}/'.format(dataset) + 'attr_file.txt', 'a', encoding='utf-8')
            fw_netflow = open('./dataset/{}/'.format(dataset) + 'attr_netflow.txt', 'a', encoding='utf-8')
            fw_memory = open('./dataset/{}/'.format(dataset) + 'attr_memory.txt', 'a', encoding='utf-8')
            fw_unnamed = open('./dataset/{}/'.format(dataset) + 'attr_unnamed.txt', 'a', encoding='utf-8')
            fw_event = open('./dataset/{}/'.format(dataset) + 'attr_event.txt', 'a', encoding='utf-8')
            for line in tqdm(f):
                # 这几种不需要关注 few information
                if 'com.bbn.tc.schema.avro.cdm18.Host' in line: continue
                if 'com.bbn.tc.schema.avro.cdm18.TimeMarker' in line or 'com.bbn.tc.schema.avro.cdm18.StartMarker' in line: continue
                if 'com.bbn.tc.schema.avro.cdm18.UnitDependency' in line or 'com.bbn.tc.schema.avro.cdm18.EndMarker' in line: continue
                uuid = pattern_uuid.findall(line)[0]
                record = pattern_record.findall(line)[0]
                id_entity_map[uuid] = record # all training dataset data
                if record == 'SrcSinkObject':
                    epoch = pid = fileDescriptor = 'null'
                    if len(pattern_epoch.findall(line)) > 0:
                        epoch = pattern_epoch.findall(line)[0]
                    if len(pattern_pid.findall(line)) > 0:
                        pid = pattern_pid.findall(line)[0]
                    if len(pattern_fileDescriptor.findall(line)) > 0:
                        fileDescriptor = pattern_fileDescriptor.findall(line)[0]
                    attr_src = str(uuid) + '\t' + str(record) + '\t' + str(epoch) + '\t' + str(
                        pid) + '\t' + str(fileDescriptor) + '\n'
                    fw_src.write(attr_src)
                if record == 'Principal':
                    user_id = euid = group_ids = principal_type = 'null'
                    if len(pattern_user_id.findall(line)) > 0:
                        user_id = pattern_user_id.findall(line)[0]
                    if len(pattern_group_ids.findall(line)) > 0:
                        group_ids = list(pattern_group_ids.findall(line)[0])
                    if len(pattern_euid.findall(line)) > 0:
                        euid = pattern_euid.findall(line)[0]
                    if len(pattern_type.findall(line)) > 0:
                        principal_type = pattern_type.findall(line)[0]
                    attr_principal = str(uuid) + '\t' + str(record) + '\t' + str(principal_type) + '\t' + str(
                        user_id) + '\t' + str(group_ids) + '\t' + str(euid) + '\n'
                    fw_principal.write(attr_principal)
                if record == 'Subject':
                    subject_type = parent = local_principal = cid = start_time = unit_id = cmdline = 'null'
                    if len(pattern_type.findall(line)) > 0:
                        subject_type = pattern_type.findall(line)[0]
                    if len(pattern_parent.findall(line)) > 0:
                        parent = pattern_parent.findall(line)[0]
                    if len(pattern_local_principal.findall(line)) > 0:
                        local_principal = pattern_local_principal.findall(line)[0]
                    if len(pattern_cid.findall(line)) > 0:
                        cid = pattern_cid.findall(line)[0]
                    if len(pattern_start_time.findall(line)) > 0:
                        start_time = pattern_start_time.findall(line)[0]
                    if len(pattern_unit_id.findall(line)) > 0:
                        unit_id = pattern_unit_id.findall(line)[0]
                    if len(pattern_cmdline.findall(line)) > 0:
                        cmdline = pattern_cmdline.findall(line)[0]
                    attr_subject = str(uuid) + '\t' + str(record) + '\t' + str(subject_type) + '\t' + str(
                        parent) + '\t' + str(
                        local_principal) + '\t' + str(cid) + '\t' + str(start_time) + '\t' + str(unit_id) + '\t' + str(
                        cmdline) + '\n'
                    fw_subject.write(attr_subject)
                if record == 'FileObject':
                    file_type = epoch = permission = path = 'null'
                    if len(pattern_type.findall(line)) > 0:
                        file_type = pattern_type.findall(line)[0]
                    if len(pattern_epoch.findall(line)) > 0:
                        epoch = pattern_epoch.findall(line)[0]
                    if len(pattern_permission.findall(line)) > 0:
                        permission = pattern_permission.findall(line)[0]
                    if len(pattern_path.findall(line)) > 0:
                        path = pattern_path.findall(line)[0]
                    attr_file = str(uuid) + '\t' + str(record) + '\t' + str(file_type) + '\t' + str(
                        epoch) + '\t' + str(permission) + '\t' + str(path) + '\n'
                    fw_file.write(attr_file)
                if record == 'NetFlowObject':
                    epoch = local_address = local_port = remote_address = remote_port = ip_protocol = 'null'
                    if len(pattern_epoch.findall(line)) > 0:
                        epoch = pattern_epoch.findall(line)[0]
                    if len(pattern_local_address.findall(line)) > 0:
                        local_address = pattern_local_address.findall(line)[0]
                    if len(pattern_local_port.findall(line)) > 0:
                        local_port = pattern_local_port.findall(line)[0]
                    if len(pattern_remote_address.findall(line)) > 0:
                        remote_address = pattern_remote_address.findall(line)[0]
                    if len(pattern_remote_port.findall(line)) > 0:
                        remote_port = pattern_remote_port.findall(line)[0]
                    if len(pattern_ip_protocol.findall(line)) > 0:
                        ip_protocol = pattern_ip_protocol.findall(line)[0]
                    attr_netflow = str(uuid) + '\t' + str(record) + '\t' + str(epoch) + '\t' + str(
                        local_address) + '\t' + str(local_port) + '\t' + str(remote_address) + '\t' + str(
                        remote_port) + '\t' + str(ip_protocol) + '\n'
                    fw_netflow.write(attr_netflow)
                if record == 'MemoryObject':
                    epoch = memory_address = tgid = size = 'null'
                    if len(pattern_epoch.findall(line)) > 0:
                        epoch = pattern_epoch.findall(line)[0]
                    if len(pattern_memory_address.findall(line)) > 0:
                        memory_address = pattern_memory_address.findall(line)[0]
                    if len(pattern_tgid.findall(line)) > 0:
                        tgid = pattern_tgid.findall(line)[0]
                    if len(pattern_size.findall(line)) > 0:
                        size = pattern_size.findall(line)[0]
                    attr_memory = str(uuid) + '\t' + str(record) + '\t' + str(epoch) + '\t' + str(
                        memory_address) + '\t' + str(tgid) + '\t' + str(size) + '\n'
                    fw_memory.write(attr_memory)
                if record == 'UnnamedPipeObject':
                    epoch = pid = source_file_descriptor = sink_file_descriptor = 'null'
                    if len(pattern_epoch.findall(line)) > 0:
                        epoch = pattern_epoch.findall(line)[0]
                    if len(pattern_pid.findall(line)) > 0:
                        pid = pattern_pid.findall(line)[0]
                    if len(pattern_sourceFileDescriptor.findall(line)) > 0:
                        source_file_descriptor = pattern_sourceFileDescriptor.findall(line)[0]
                    if len(pattern_sinkFileDescriptor.findall(line)) > 0:
                        sink_file_descriptor = pattern_sinkFileDescriptor.findall(line)[0]
                    attr_unnamed = str(uuid) + '\t' + str(record) + '\t' + str(epoch) + '\t' + str(
                        pid) + '\t' + str(source_file_descriptor) + '\t' + str(sink_file_descriptor) + '\n'
                    fw_unnamed.write(attr_unnamed)
                if record == 'Event':
                    event_type = seq = thread_id = src = size = dst1 = dst2 = time = 'null'
                    if len(pattern_type.findall(line)) > 0:
                        event_type = pattern_type.findall(line)[0]
                    if len(pattern_seq.findall(line)) > 0:
                        seq = pattern_seq.findall(line)[0]
                    if len(pattern_thread_id.findall(line)) > 0:
                        thread_id = pattern_thread_id.findall(line)[0]
                    if len(pattern_src.findall(line)) > 0:
                        src = pattern_src.findall(line)[0]
                    if len(pattern_size.findall(line)) > 0:
                        size = pattern_size.findall(line)[0]
                    if len(pattern_dst1.findall(line)) > 0:
                        dst1 = pattern_dst1.findall(line)[0]
                    if len(pattern_dst2.findall(line)) > 0:
                        dst2 = pattern_dst2.findall(line)[0]
                    if len(pattern_time.findall(line)) > 0:
                        time = pattern_time.findall(line)[0]
                    # if 'READ' in event_type or 'RECV' in event_type or 'LOAD' in event_type:
                    attr_event = str(uuid) + '\t' + str(record) + '\t' + str(event_type) + '\t' + str(
                        seq) + '\t' + str(thread_id) + '\t' + str(src) + '\t' + str(dst1) + '\t' + str(
                        dst2) + '\t' + str(size) + '\t' + str(time) + '\n'
                    fw_event.write(attr_event)
            fw_src.close()
            fw_principal.close()
            fw_subject.close()
            fw_file.close()
            fw_netflow.close()
            fw_memory.close()
            fw_unnamed.close()
            fw_event.close()
    if len(id_entity_map) != 0:
        fw_id_entity_map = open('./dataset/{}/'.format(dataset) + 'id_entity_map.json', 'w', encoding='utf-8')
        json.dump(id_entity_map, fw_id_entity_map)
        fw_id_entity_map.close()


# 找出所有的节点对
def find_entity_pair(dataset):
    record_cnt_map = {}  # 记录event,src和dst这些的数字编号(uuid,cnt)
    entity_cnt = 0
    if os.path.exists('./dataset/{}/id_entity_map.json'.format(dataset)):
        with open('./dataset/{}/id_entity_map.json'.format(dataset), 'r', encoding='utf-8') as f_id_entity_map:
            id_entity_map = json.load(f_id_entity_map)
    for file in os.listdir('./dataset/{}/'.format(dataset)):
        if 'event' in file:   # attr_event.txt
            fw_entity_pair = open('./dataset/{}/'.format(dataset) + 'entity_pair.txt', 'w', encoding='utf-8')
            print('searching behavior entity pair {} ...'.format(file))
            f = open('./dataset/{}/'.format(dataset) + 'attr_event.txt', 'r', encoding='utf-8')
            for l in f.readlines():
                split_line = l.split('\t')
                uuid, record, event_type, seq, thread_id, src, dst1, dst2, size, time = split_line
                if uuid not in record_cnt_map:
                    record_cnt_map[uuid] = entity_cnt
                    entity_cnt += 1
                if src in id_entity_map and dst1 in id_entity_map:
                    if src not in record_cnt_map:
                        record_cnt_map[src] = entity_cnt
                        entity_cnt += 1
                    if dst1 not in record_cnt_map:
                        record_cnt_map[dst1] = entity_cnt
                        entity_cnt += 1
                    entity_pair = str(record_cnt_map[uuid]) + '\t' + str(record_cnt_map[src]) + '\t' + str(
                        record_cnt_map[dst1]) + '\n'
                    fw_entity_pair.write(entity_pair)
                    # entity_pair 存放的是 event src dst 的 编号
                    # record_cnt_map 存放的是 uuid - cnt
                if dst2 in id_entity_map:
                    if dst2 not in record_cnt_map:
                        record_cnt_map[dst2] = entity_cnt
                        entity_cnt += 1
                    entity_pair = str(record_cnt_map[uuid]) + '\t' + str(record_cnt_map[src]) + '\t' + str(
                        record_cnt_map[dst2]) + '\n'
                    fw_entity_pair.write(entity_pair)

                # if 'READ' in event_type or 'RECV' in event_type or 'LOAD' in event_type:注意一下这个顺序
            fw_entity_pair.close()

            # finish the entity_pair.txt 
    if len(record_cnt_map) != 0:
        fw_record_cnt_map = open('./dataset/{}/'.format(dataset) + 'record_cnt_map.json', 'w', encoding='utf-8')
        json.dump(record_cnt_map, fw_record_cnt_map)
        fw_record_cnt_map.close()
        # finish the record_map.txt
        cnt_record_map= {str(v): k for k, v in record_cnt_map.items()}
        # reverse
        fw_cnt_record_map= open('./dataset/{}/'.format(dataset) + 'cnt_record_map.json', 'w', encoding='utf-8')
        json.dump(cnt_record_map, fw_cnt_record_map)
        fw_cnt_record_map.close()
        # finish the count_record.txt
# event_list={{},..,{}}

def generate_graphs_in_batches(event_list,cnt_record_map, uuid_to_node_attrs, id_entity_map, batch_size=1000):
    for batch_start in tqdm(range(0, len(event_list), batch_size), desc='Processing batches'):
        batch_event_pairs = event_list[batch_start:batch_start + batch_size]  # 本批次需要处理的event
        batch_graphs = []
        batch_embeddings = []
        new_edges_dict = {}
        whole_G = nx.Graph()
        for event_pair in batch_event_pairs:
            G = nx.Graph()
            event_uuid = event_pair['event']
            src_uuid = event_pair['src']
            dst_uuid = event_pair['dst']
            print(event_uuid)
            print(src_uuid)
            uuids = [event_uuid,src_uuid, dst_uuid ]
            # 构建节点和边字典
            nodes_dict = {}
            edges_dict = {}
            cnt_node = 0
            cnt_edge = 0
            cnt_src_num = 0
            src_record = id_entity_map[src_uuid]
            dst_record = id_entity_map[dst_uuid]
            for _ in uuids:
            # 首先处理Subject - Subject的节点对
                if src_record == 'Subject' and dst_record== 'Subject':
                    node_attrs = uuid_to_node_attrs[_]
                    # 添加节点和该uuid的连接
                    for attr_name, attr_value in node_attrs.items():
                        NODE_ATTR = 9
                        if cnt_node % NODE_ATTR == 0:
                            new_node_dict[cnt_node / NODE_ATTR] = {attr_name: attr_value}
                        nodes_dict[cnt_node] = {attr_name: attr_value}
                        cnt_node += 1

                        # if cnt_node <= 5:
                        #     # 插入打印结果
                        #     print(f"属性名称: {attr_name}, 属性值: {attr_value}")
                        #     print(f"当前节点索引: {cnt_node}, 更新后的 nodes_dict: {nodes_dict}")
                        #     continue
                        # assert 1==0

                    # 建立该节点与属性节点的边
                    for i in range(cnt_src_num + 1, cnt_node):
                        edges_dict[(cnt_src_num, i)] = None
            # TODO 建立每一个节点之间联系的边
            for event_pair2 in batch_event_pairs:
                event_uuid2 = event_pair['event']
                src_uuid2 = event_pair['src']
                ## 1 2 3
                ## 4 3 5
 
                ##
                if dst_uuid == src_uuid2:
                    new_edges_dict[cnt_edge] = {event_uuid,event_uuid2}
                    cnt_edge +=1
            print(nodes_dict.items())
            G.add_nodes_from(nodes_dict.items())
            G.add_edges_from(edges_dict.keys())
            whole_G.add_nodes_from(new_node_dict.items())
            whole_G.add_edges_from(new_edges_dict.keys())
            # 编码和池化聚合
            node_features = torch.tensor([list(v.values()) for v in nodes_dict.values()], dtype=torch.float)
            edge_index = torch.tensor([[k[0], k[1]] for k in edges_dict.keys()], dtype=torch.long).t()

            encoder = GCNEncoder(node_features.shape[1], 32, 64)
            graph_embedding = global_mean_pool(encoder(node_features, edge_index), torch.tensor([0] * len(nodes_dict)))

            batch_graphs.append(G)
            batch_embeddings.append(graph_embedding)
    


def test(dataset, record_cnt_map=None, entity_pair=None):
    entity_pair_counts = {}
    record_cnt_map = {}
    if os.path.exists('./dataset/{}/id_entity_map.json'.format(dataset)):
        with open('./dataset/{}/id_entity_map.json'.format(dataset), 'r', encoding='utf-8') as f_id_entity_map:
            print('loading id_entity_map')
            id_entity_map = json.load(f_id_entity_map)
    if os.path.exists('./dataset/{}/record_cnt_map.json'.format(dataset)):
        with open('./dataset/{}/record_cnt_map.json'.format(dataset), 'r', encoding='utf-8') as f_record_cnt_map:
            record_cnt_map = json.load(f_record_cnt_map)
    record_cnt_map_tmp = {str(v): k for k, v in record_cnt_map.items()}
    if os.path.exists('./dataset/{}/entity_pair.txt'.format(dataset)):
        with open('./dataset/{}/entity_pair.txt'.format(dataset), 'r', encoding='utf-8') as f:
            for line in f.readlines():
                event = line.strip().split('\t')
                entity_pair = {
                    'event_record': record_cnt_map_tmp[event[0]],
                    'src_record': record_cnt_map_tmp[event[1]],
                    'dst_record': record_cnt_map_tmp[event[2]]
                }
                record_type = id_entity_map[entity_pair['event_record']] + ' ' + id_entity_map[
                    entity_pair['src_record']] + ' ' + id_entity_map[entity_pair['dst_record']]
                if record_type not in entity_pair_counts:
                    entity_pair_counts[record_type] = 1
                else:
                    entity_pair_counts[record_type] += 1

    for entity_pair, count in entity_pair_counts.items():
        print(f"entity_pair: {entity_pair}, count: {count}")

# 构建节点属性图
def graph_node_construction(dataset,G,uuid_to_node_attrs):
    subject_attr_num = 0
    file_attr_num = 0
    network_attr_num = 0
    mem_attr_num = 0
    SUBJECT_ATTR = 9
    FILE_ATTR = 6
    NETWORK_ATTR = 8
    MEM_ATTR = 6
    if os.path.exists('./dataset/{}/attr_subject.txt'.format(dataset)):
        with open('./dataset/{}/attr_subject.txt'.format(dataset), 'r', encoding='utf-8') as f_attr_sub:
            for attr in tqdm(f_attr_sub.readlines()):
                attr_values = attr.strip().split('\t')
                node_attrs = {
                    'uuid': attr_values[0],
                    'record': attr_values[1],
                    'subject_type': attr_values[2],
                    'parent': attr_values[3],
                    'local_principal': attr_values[4],
                    'cid': attr_values[5],
                    'start_time': attr_values[6],
                    'unit_id': attr_values[7],
                    'cmdline': attr_values[8]
                }
                subject_attr_num += SUBJECT_ATTR
                uuid = node_attrs['uuid']
                uuid_to_node_attrs[uuid] = node_attrs
                G.add_node(node_attrs['uuid'], **node_attrs)


    if os.path.exists('./dataset/{}/attr_file.txt'.format(dataset)):
        with open('./dataset/{}/attr_file.txt'.format(dataset), 'r', encoding='utf-8') as fw_file_sub:
            for attr in tqdm(fw_file_sub.readlines()):
                attr_values = attr.strip().split('\t')
                # uuid	record	file_type	epoch	permission	path
                node_attrs = {                                 
                    'uuid': attr_values[0],
                    'record': attr_values[1],
                    'file_type': attr_values[2],
                    'epoch': attr_values[3],
                    'permission': attr_values[4],
                    'path': attr_values[5]
                }
                file_attr_num += FILE_ATTR
                uuid = node_attrs['uuid']
                uuid_to_node_attrs[uuid] = node_attrs
                G.add_node(node_attrs['uuid'], **node_attrs)


    if os.path.exists('./dataset/{}/attr_network.txt'.format(dataset)):
        with open('./dataset/{}/attr_network.txt'.format(dataset), 'r', encoding='utf-8') as fw_network_sub:
            for attr in tqdm(fw_network_sub.readlines()):
                attr_values = attr.strip().split('\t')
                # uuid	record	epoch	local_address	local_port	remote_address	remote_port	ip_protocol
                node_attrs = {                                 
                    'uuid': attr_values[0],
                    'record': attr_values[1],
                    'epoch': attr_values[2],
                    'local_address': attr_values[3],
                    'local_port': attr_values[4],
                    'remote_address': attr_values[5],
                    'remote_port': attr_values[6],
                    'ip_protocol': attr_values[7]
                }
                network_attr_num += NETWORK_ATTR
                uuid = node_attrs['uuid']
                uuid_to_node_attrs[uuid] = node_attrs
                G.add_node(node_attrs['uuid'], **node_attrs)

    
    if os.path.exists('./dataset/{}/attr_mem.txt'.format(dataset)):
        with open('./dataset/{}/attr_mem.txt'.format(dataset), 'r', encoding='utf-8') as f_mem_sub:
            for attr in tqdm(f_mem_sub.readlines()):
                attr_values = attr.strip().split('\t')
                # uuid	record	epoch	memory_address	tgid	size
                node_attrs = {
                    'uuid': attr_values[0],
                    'record': attr_values[1],
                    'epoch': attr_values[2],
                    'memory_address': attr_values[3],
                    'tgid': attr_values[4],
                    'size': attr_values[5]
                }
                mem_attr_num += MEM_ATTR
                uuid = node_attrs['uuid']
                uuid_to_node_attrs[uuid] = node_attrs
                G.add_node(node_attrs['uuid'], **node_attrs)
    return




def graph_edge_construction(dataset,G,uuid_to_node_attrs,event_list):
    edges_dict = {}
    if os.path.exists('./dataset/{}/id_entity_map.json'.format(dataset)):
        with open('./dataset/{}/id_entity_map.json'.format(dataset), 'r', encoding='utf-8') as f_id_entity_map:
            print('loading id_entity_map')
            id_entity_map = json.load(f_id_entity_map)
    if os.path.exists('./dataset/{}/cnt_record_map.json'.format(dataset)):
        with open('./dataset/{}/cnt_record_map.json'.format(dataset), 'r', encoding='utf-8') as f_cnt_record_map:
            print('loading cnt_record_map')
            cnt_record_map = json.load(f_cnt_record_map)
    if os.path.exists('./dataset/{}/entity_pair.txt'.format(dataset)):
        with open('./dataset/{}/entity_pair.txt'.format(dataset), 'r', encoding='utf-8') as f:
            print('loading event_list')
            for line in f.readlines():
                event = line.strip().split('\t')
                entity_pair ={
                    'event': cnt_record_map[event[0]],
                    'src': cnt_record_map[event[1]],
                    'dst': cnt_record_map[event[2]],
                }
                attr_dict = ['Event','Subject','FileObject','NetFlowObject','MemoryObject']
                if id_entity_map[entity_pair['src']] in attr_dict and id_entity_map[entity_pair['dst']] in attr_dict:
                    edges_dict[(entity_pair['src'], entity_pair['dst'])] = None
    G.add_edges_from(edges_dict.keys())
    return
# 对每个节点对进行处理，构建子图
# 从event读出src和dst,匹配两者类型，去相应文件读取属性，将节点相连
# 先尝试建立所有subject事件对
def graph_construction(dataset):
    G = nx.Graph()
    uuid_to_node_attrs = {}
    event_list = []
    graph_list = []
    graph_node_construction(dataset,G,uuid_to_node_attrs)
    graph_edge_construction(dataset,G,uuid_to_node_attrs,event_list)
    # 创建好正常图
    graph_generator = generate_graphs_in_batches(event_list, cnt_record_map,uuid_to_node_attrs, id_entity_map)
    graph_embedding_list = []
    for batch_graphs, batch_embeddings in tqdm(graph_generator, total=len(event_list) // 1000,
                                               desc='Collecting graphs'):
        graph_list.extend(batch_graphs)
        graph_embedding_list.extend(batch_embeddings)
    print("First 5 graphs in graph_list:")
    for graph in graph_list[:5]:
        print(graph)

    print("\nFirst 5 embeddings in graph_embedding_list:")
    for embedding in graph_embedding_list[:5]:
        print(embedding)
    return graph_list, graph_embedding_list




def count_frequency(numbers):
    return collections.Counter(numbers)


normal_num = 0


# 辅助find_bining_max_value中绘图判定
def is_malicious(tgid):
    global normal_num
    if tgid > 0:
        normal_num += 1
        return True
    else:
        return False


# 辅助find_bining_max_value中绘图
def plot_distribution(counter, title, dataset, path, is_malicious=None):
    labels, values = zip(*counter.items())

    # 排序
    labels, values = zip(*sorted(zip(labels, values)))
    colors = ['r' if is_malicious and is_malicious(int(l)) else 'b' for l in labels]

    plt.figure(figsize=(10, 5))
    plt.scatter(labels, values, c=colors)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    # plt.yscale('log')  # 使用对数刻度（如果值的范围很大）
    plt.title(title)
    plt.savefig('./dataset/{}/{}'.format(dataset, path))
    plt.show()


# 返回恶意uuid
def malicious(dataset):
    malicious_entities = set()
    if os.path.exists('./groundtruth/{}.txt'.format(dataset)):
        with open('./groundtruth/{}.txt'.format(dataset), 'r', encoding='utf-8') as f:
            for l in f.readlines():
                malicious_entities.add(l.lstrip().rstrip())
    return malicious_entities


# 查看正常节点属性的恶意节点属性分布
def find_bining_max_value(dataset, attr):
    global normal_num
    attr_entity_name = name_map[attr]
    malicious_entities = set()
    malicious_entities = malicious(dataset)
    if os.path.exists('./dataset/{}/attr_{}.txt'.format(dataset, attr_entity_name)):
        with open('./dataset/{}/attr_{}.txt'.format(dataset, attr_entity_name), 'r', encoding='utf-8') as f_attr:
            if attr_entity_name == 'memory':
                size_list = []
                memory_ad_list = []
                tgid_list = []
                cnt = 0
                malicious_set = set()
                for line in f_attr:
                    fields = line.strip().split('\t')
                    uuid = fields[0]
                    memory_ad = int(fields[3])
                    tgid = int(fields[4])
                    size = int(fields[5])
                    if uuid in malicious_entities:
                        print('tgid={}'.format(tgid))
                        tgid = -tgid
                        memory_ad = -memory_ad
                        size = -size
                        cnt += 1
                    size_list.append(size)
                    memory_ad_list.append(memory_ad)
                    tgid_list.append(tgid)
                size_counter = count_frequency(size_list)
                memory_ad_counter = count_frequency(memory_ad_list)
                tgid_counter = count_frequency(tgid_list)
                # colors = ['b' if not is_malicious(int(t), malicious_entities) else 'r' for t in x]
                plot_distribution(size_counter, 'SIZE Distribution', dataset, 'memory_size.png', is_malicious)
                plot_distribution(memory_ad_counter, 'MEMORY_AD Distribution', dataset, 'memory_ad.png', is_malicious)
                plot_distribution(tgid_counter, "TGID Distribution", dataset, 'tgid.png', is_malicious)
                print(normal_num)
                print('cnt={}'.format(cnt))
            if attr_entity_name == 'src':
                # uuid record epoch pid fileDescriptor
                epoch_list = []
                pid_list = []
                file_descriptor_list = []
                cnt = 0
                for line in f_attr:
                    fields = line.strip().split('\t')
                    uuid = fields[0]
                    epoch = int(fields[2])
                    pid = int(fields[3])
                    file_descriptor = int(fields[4])
                    if uuid in malicious_entities:
                        epoch = -epoch
                        pid = -pid
                        file_descriptor = -file_descriptor
                        cnt += 1
                    epoch_list.append(epoch)
                    pid_list.append(pid)
                    file_descriptor_list.append(file_descriptor)
                epoch_counter = count_frequency(epoch_list)
                pid_counter = count_frequency(pid_list)
                file_descriptor_counter = count_frequency(file_descriptor_list)
                plot_distribution(epoch_counter, 'EPOCH Distribution', dataset, 'epoch.png', is_malicious)
                plot_distribution(pid_counter, 'PID Distribution', dataset, 'pid.png', is_malicious)
                plot_distribution(file_descriptor_counter, "FILE_DESCRIPTOR Distribution", dataset,
                                  'file_descriptor.png', is_malicious)
                print('cnt={}'.format(cnt))
            if attr_entity_name == 'event':
                # uuid record event_type seq thread_id src dst1 dst2 size time
                seq_list = []
                thread_id_list = []
                size_list = []
                time_list = []
                for line in f_attr:
                    fields = line.strip().split('\t')
                    seq = int(fields[3])
                    seq_list.append(seq)
                seq_counter = count_frequency(seq_list)
                plot_distribution(seq_counter, 'SEQ Distribution')

            return 1
    else:
        raise NotImplementedError("It's not exist")
        return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Darpa TC E3 Parser')
    parser.add_argument("--dataset", type=str, default="trace")
    args = parser.parse_args()
    dataset = args.dataset
    if dataset not in ['trace', 'theia', 'cadets']:
        raise NotImplementedError("This dataset is not included")

    # preprocess(dataset)
    # find_entity_pair(dataset)
    # test(dataset)
    graph_construction(dataset)
    # print(find_bining_max_value(dataset, 'SrcSinkObject'))
