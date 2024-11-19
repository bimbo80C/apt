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
import pandas as pd
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
        if 'json' in file and 'entity' not in file  and '5' not in file and '6' not in file:
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


#测试
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


# 构建节点属性图,并且将子图以uuid的形式加入整图G当中
# def get_attrs(dataset,uuid_to_node_attrs,uuid_to_edge_attrs):
#     # entity == subject
#     f_node = open('./dataset/{}/'.format(dataset) + 'uuid_to_node_attrs.txt', 'a', encoding='utf-8')
#     f_edge = open('./dataset/{}/'.format(dataset) + 'uuid_to_edge_attrs.txt', 'a', encoding='utf-8')
#     if os.path.exists('./dataset/{}/attr_subject.txt'.format(dataset)):
#         with open('./dataset/{}/attr_subject.txt'.format(dataset), 'r', encoding='utf-8') as f_attr_sub:
#             for attr in tqdm(f_attr_sub.readlines()):
#                 attr_values = attr.strip().split('\t')
#                 node_attrs = {
#                     'uuid': attr_values[0],
#                     'record': attr_values[1],
#                     'subject_type': attr_values[2],
#                     'parent': attr_values[3],
#                     'local_principal': attr_values[4],
#                     'cid': attr_values[5],
#                     'start_time': attr_values[6],
#                     'unit_id': attr_values[7],
#                     'cmdline': attr_values[8]
#                 }
#                 uuid = node_attrs['uuid']
#                 # uuid_to_node_attrs[uuid] = node_attrs
#                 json.dump(node_attrs, f_node)
#
#     if os.path.exists('./dataset/{}/attr_file.txt'.format(dataset)):
#         with open('./dataset/{}/attr_file.txt'.format(dataset), 'r', encoding='utf-8') as fw_file_sub:
#             for attr in tqdm(fw_file_sub.readlines()):
#                 attr_values = attr.strip().split('\t')
#                 # uuid	record	file_type	epoch	permission	path
#                 node_attrs = {
#                     'uuid': attr_values[0],
#                     'record': attr_values[1],
#                     'file_type': attr_values[2],
#                     'epoch': attr_values[3],
#                     'permission': attr_values[4],
#                     'path': attr_values[5]
#                 }
#                 uuid = node_attrs['uuid']
#                 uuid_to_node_attrs[uuid] = node_attrs
#     if os.path.exists('./dataset/{}/attr_netflow.txt'.format(dataset)):
#         with open('./dataset/{}/attr_netflow.txt'.format(dataset), 'r', encoding='utf-8') as fw_network_sub:
#             for attr in tqdm(fw_network_sub.readlines()):
#                 attr_values = attr.strip().split('\t')
#                 # uuid	record	epoch	local_address	local_port	remote_address	remote_port	ip_protocol
#                 node_attrs = {
#                     'uuid': attr_values[0],
#                     'record': attr_values[1],
#                     'epoch': attr_values[2],
#                     'local_address': attr_values[3],
#                     'local_port': attr_values[4],
#                     'remote_address': attr_values[5],
#                     'remote_port': attr_values[6],
#                     'ip_protocol': attr_values[7]
#                 }
#                 uuid = node_attrs['uuid']
#                 uuid_to_node_attrs[uuid] = node_attrs
#     if os.path.exists('./dataset/{}/attr_memory.txt'.format(dataset)):
#         with open('./dataset/{}/attr_memory.txt'.format(dataset), 'r', encoding='utf-8') as f_mem_sub:
#             for attr in tqdm(f_mem_sub.readlines()):
#                 attr_values = attr.strip().split('\t')
#                 # uuid	record	epoch	memory_address	tgid	size
#                 node_attrs = {
#                     'uuid': attr_values[0],
#                     'record': attr_values[1],
#                     'epoch': attr_values[2],
#                     'memory_address': attr_values[3],
#                     'tgid': attr_values[4],
#                     'size': attr_values[5]
#                 }
#                 uuid = node_attrs['uuid']
#                 uuid_to_node_attrs[uuid] = node_attrs
#     if os.path.exists('./dataset/{}/attr_event.txt'.format(dataset)):
#         with open('./dataset/{}/attr_event.txt'.format(dataset), 'r', encoding='utf-8') as f_event_sub:
#             for attr in tqdm(f_event_sub.readlines()):
#                 attr_values = attr.strip().split('\t')
#                 # uuid	record	event_type	seq	thread_id	src	dst1	dst2	size	time
#                 edge_attrs = {
#                     'record': attr_values[1],
#                     'event_type': attr_values[2],
#                     'seq': attr_values[3],
#                     'thread_id': attr_values[4],
#                     'size': attr_values[8],
#                     'time': attr_values[9],
#                 }
#                 uuid = attr_values[0]
#                 uuid_to_edge_attrs[uuid] = edge_attrs
#     return

def get_attrs(dataset):
    # entity == subject
    uuid_to_node_attrs={}
    uuid_to_edge_attrs={}
    if os.path.exists('./dataset/{}/attr_subject.txt'.format(dataset)):
        with open('./dataset/{}/attr_subject.txt'.format(dataset), 'r', encoding='utf-8') as f_sub:
            df = pd.read_csv(f_sub,
                             sep='\t',
                             names=['uuid', 'record', 'subject_type', 'parent',
                                    'local_principal', 'cid', 'start_time',
                                    'unit_id', 'cmdline'])
            uuid_to_node_attrs.update(df.set_index('uuid').to_dict('index'))

    if os.path.exists('./dataset/{}/attr_file.txt'.format(dataset)):
        with open('./dataset/{}/attr_file.txt'.format(dataset), 'r', encoding='utf-8') as f_file:
            df = pd.read_csv(f_file,
                             sep='\t',
                             names=['uuid', 'record', 'file_type', 'epoch',
                                    'permission','path'])
            uuid_to_node_attrs.update(df.set_index('uuid').to_dict('index'))

    if os.path.exists('./dataset/{}/attr_netflow.txt'.format(dataset)):
        with open('./dataset/{}/attr_netflow.txt'.format(dataset), 'r', encoding='utf-8') as f_netflow:
            df = pd.read_csv(f_netflow,
                             sep='\t',
                             names=['uuid', 'record', 'epoch',
                                    'local_address',
                                    'local_port',
                                    'remote_address',
                                    'remote_port',
                                    'ip_protocol'])
            uuid_to_node_attrs.update(df.set_index('uuid').to_dict('index'))

    if os.path.exists('./dataset/{}/attr_memory.txt'.format(dataset)):
        with open('./dataset/{}/attr_memory.txt'.format(dataset), 'r', encoding='utf-8') as f_mem:
            df = pd.read_csv(f_mem,
                             sep='\t',
                             names=['uuid', 'record', 'epoch',
                                    'memory_address',
                                    'tgid',
                                    'size'])
            uuid_to_node_attrs.update(df.set_index('uuid').to_dict('index'))


    if os.path.exists('./dataset/{}/attr_event.txt'.format(dataset)):
        with open('./dataset/{}/attr_event.txt'.format(dataset), 'r', encoding='utf-8') as f_event:
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
    return uuid_to_node_attrs,uuid_to_edge_attrs

def get_maps(dataset):
    if os.path.exists('./dataset/{}/id_entity_map.json'.format(dataset)):
        with open('./dataset/{}/id_entity_map.json'.format(dataset), 'r', encoding='utf-8') as f_id_entity_map:
            print('loading id_entity_map')
            id_entity_map = json.load(f_id_entity_map)
    if os.path.exists('./dataset/{}/cnt_record_map.json'.format(dataset)):
        with open('./dataset/{}/cnt_record_map.json'.format(dataset), 'r', encoding='utf-8') as f_cnt_record_map:
            print('loading cnt_record_map')
            cnt_record_map = json.load(f_cnt_record_map)
    return id_entity_map, cnt_record_map
def graph_node_construction(dataset,uuid_to_node_attrs,uuid_to_edge_attrs,id_entity_map,cnt_record_map):
    if os.path.exists('./dataset/{}/entity_pair.txt'.format(dataset)):
        with open('./dataset/{}/entity_pair.txt'.format(dataset), 'r', encoding='utf-8') as f:
            print('loading event_list for node_construction')
            for line in f:
                event = line.strip().split('\t')
                entity_pair ={
                    'event': cnt_record_map[event[0]],
                    'src': cnt_record_map[event[1]],
                    'dst': cnt_record_map[event[2]],
                }
                attr_dict = ['Event','Subject','FileObject','NetFlowObject','MemoryObject']
                if id_entity_map[entity_pair['src']] in attr_dict and id_entity_map[entity_pair['dst']] in attr_dict:
                    event_uuid = entity_pair['event']
                    src_uuid = entity_pair['src']
                    dst_uuid = entity_pair['dst']
                    event_attr = uuid_to_edge_attrs[event_uuid]
                    src_attr = uuid_to_node_attrs[src_uuid]
                    dst_attr = uuid_to_node_attrs[dst_uuid]
                    sub_G = nx.DiGraph()
                    sub_nodes_dict = {}
                    sub_edges_dict = {}
                    cnt_node = 0
                    for attr_name,attr_value in src_attr.items():
                        sub_nodes_dict[cnt_node] = attr_name + str(':') + str(attr_value)
                        cnt_node +=1
                        if attr_name != 'uuid':
                            sub_edges_dict[(str('uuid:')+src_uuid, attr_name + str(':') + str(attr_value))] = None
                        
                    for attr_name, attr_value in dst_attr.items():
                        sub_nodes_dict[cnt_node] = attr_name + str(':') + attr_value # 形式需要确认下20241107
                        cnt_node +=1
                        if attr_name != 'uuid':
                            sub_edges_dict[(str('uuid:')+dst_uuid, attr_name + str(':') + str(attr_value))] = None
                    sub_edges_dict[(str('uuid:')+src_uuid,str('uuid:')+dst_uuid)] = event_attr # add edge (src,dst)
                    sub_G.add_nodes_from(sub_nodes_dict.values())
                    sub_G.add_edges_from(sub_edges_dict.keys())
                    graph_list.append(sub_G)
    return graph_list

# 待更新
def graph_node_embedding(graph_list):
    for graph in graph_list:
        for node in graph.nodes():
            print(node)
    return

# 将节点对整体之外的边添加到图中
def graph_edge_construction(dataset,edges_set,cnt_record_map):
    if os.path.exists('./dataset/{}/entity_pair.txt'.format(dataset)):
        with open('./dataset/{}/entity_pair.txt'.format(dataset), 'r', encoding='utf-8') as f:
            print('loading event_list for edge_construction')
            lines = f.readlines()
            for i in range(len(lines)):
                line_i = lines[i].strip().split('\t') 
                for j in range(i + 1, len(lines)):
                    line_j = lines[j].strip().split('\t')
                    if line_i[2] == line_j[1]:
                        edges_set.add((cnt_record_map[line_i[0]], cnt_record_map[line_j[0]]))
    return
# 对每个节点对进行处理，构建子图
# 从event读出src和dst,匹配两者类型，去相应文件读取属性，将节点相连
# 先尝试建立所有subject事件对
def graph_construction(dataset):
    G = nx.Graph()
    edges_set = set()
    # graph_list = []
    graph_embedding_list = []
    # uuid_to_node_attrs = {}
    # uuid_to_edge_attrs = {}
    # id_entity_map = {}
    # cnt_record_map = {}
    uuid_to_node_attrs,uuid_to_edge_attrs = get_attrs(dataset)
    id_entity_map,cnt_record_map = get_maps(dataset)
    graph_list = graph_node_construction(dataset,uuid_to_node_attrs,uuid_to_edge_attrs,id_entity_map,cnt_record_map)
    graph_node_embedding(graph_list)
    graph_edge_construction(dataset,edges_set,cnt_record_map)
    G.add_edges_from(edges_set)
    return graph_list, graph_embedding_list

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

