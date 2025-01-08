import os
import argparse
from tqdm import tqdm
import re
import json
import networkx as nx
import numpy as np
import pandas as pd
import pickle as pkl

from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import dgl
import ipaddress
from sklearn.preprocessing import OneHotEncoder
from collections import defaultdict
import matplotlib.pyplot as plt
import random

metadata = {
    'trace': {
        'train': ['ta1-trace-e3-official-1.json.0', 'ta1-trace-e3-official-1.json.1', 'ta1-trace-e3-official-1.json.2',
                  'ta1-trace-e3-official-1.json.3'],
        'test': ['ta1-trace-e3-official-1.json.0', 'ta1-trace-e3-official-1.json.1', 'ta1-trace-e3-official-1.json.2',
                 'ta1-trace-e3-official-1.json.3', 'ta1-trace-e3-official-1.json.4'] # 'ta1-trace-e3-official-1.json.4'
    },
    'theia': {
        'train': ['ta1-theia-e3-official-6r.json', 'ta1-theia-e3-official-6r.json.1', 'ta1-theia-e3-official-6r.json.2',
                  'ta1-theia-e3-official-6r.json.3'],
        'test': ['ta1-theia-e3-official-6r.json.8']
    },
    'cadets': {
        'train': ['ta1-cadets-e3-official.json', 'ta1-cadets-e3-official.json.1', 'ta1-cadets-e3-official.json.2',
                  'ta1-cadets-e3-official-2.json.1'],
        'test': ['ta1-cadets-e3-official-2.json']
    }
}

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

# 先将所有实体和属性提取出来，数据集的分割按照event来
def preprocess_entity_attr(dataset):
    id_entity_map = {}
    for file in os.listdir('./dataset/{}/'.format(dataset)):  # file ta1-trace-e3-official-1.json except 5 & 6 for testing
        if 'json' in file and '5' not in file and '6' not in file and 'id' not in file:
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
                id_entity_map[uuid] = record  # all training dataset data
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
                    event_type =  src = dst1 = dst2 = time = 'null'
                    if len(pattern_type.findall(line)) > 0:
                        event_type = pattern_type.findall(line)[0]
                    if len(pattern_src.findall(line)) > 0:
                        src = pattern_src.findall(line)[0]
                    if len(pattern_dst1.findall(line)) > 0:
                        dst1 = pattern_dst1.findall(line)[0]
                    if len(pattern_dst2.findall(line)) > 0:
                        dst2 = pattern_dst2.findall(line)[0]
                    if len(pattern_time.findall(line)) > 0:
                        time = pattern_time.findall(line)[0]
                    # if 'READ' in event_type or 'RECV' in event_type or 'LOAD' in event_type:
                    attr_event = str(uuid) + '\t' + str(record) + '\t' + str(event_type)  + '\t' + str(src) + '\t' + str(dst1) + '\t' + str(
                        dst2) + '\t' + str(time) + '\n'
                    fw_event.write(attr_event)
            fw_src.close()
            fw_principal.close()
            fw_subject.close()
            fw_file.close()
            fw_netflow.close()
            fw_memory.close()
            fw_unnamed.close()
            fw_event.close()
            # 从每一个文件当中提取字段
    if len(id_entity_map) != 0:
        fw_id_entity_map = open('./dataset/{}/'.format(dataset) + 'id_entity_map.json', 'w', encoding='utf-8')
        json.dump(id_entity_map, fw_id_entity_map)
        fw_id_entity_map.close()

###############
# 提取训练集和测试集的事件
###############
# def preprocess(dataset, mode):
#     for file in os.listdir('./dataset/{}/'.format(dataset)):
#         if file in metadata[dataset][mode]:
#             print('reading event{} ...'.format(file))
#             f = open('./dataset/{}/'.format(dataset) + file, 'r', encoding='utf-8')
#             fw_event = open('./dataset/{}/{}/'.format(dataset, mode) + 'attr_event.txt', 'a', encoding='utf-8')
#             for line in tqdm(f):
#                 uuid = pattern_uuid.findall(line)[0]
#                 record = pattern_record.findall(line)[0]
#                 if record == 'Event':
#                     event_type = seq = thread_id = src = size = dst1 = dst2 = time = 'null'
#                     if len(pattern_type.findall(line)) > 0:
#                         event_type = pattern_type.findall(line)[0]
#                     if len(pattern_seq.findall(line)) > 0:
#                         seq = pattern_seq.findall(line)[0]
#                     if len(pattern_thread_id.findall(line)) > 0:
#                         thread_id = pattern_thread_id.findall(line)[0]
#                     if len(pattern_src.findall(line)) > 0:
#                         src = pattern_src.findall(line)[0]
#                     if len(pattern_size.findall(line)) > 0:
#                         size = pattern_size.findall(line)[0]
#                     if len(pattern_dst1.findall(line)) > 0:
#                         dst1 = pattern_dst1.findall(line)[0]
#                     if len(pattern_dst2.findall(line)) > 0:
#                         dst2 = pattern_dst2.findall(line)[0]
#                     if len(pattern_time.findall(line)) > 0:
#                         time = pattern_time.findall(line)[0]
#                     # if 'READ' in event_type or 'RECV' in event_type or 'LOAD' in event_type:
#                     attr_event = str(uuid) + '\t' + str(record) + '\t' + str(event_type) + '\t' + str(
#                         seq) + '\t' + str(thread_id) + '\t' + str(src) + '\t' + str(dst1) + '\t' + str(
#                         dst2) + '\t' + str(size) + '\t' + str(time) + '\n'
#                     fw_event.write(attr_event)
#             fw_event.close()

def preprocess(dataset):
    for mode in metadata[dataset]:
        for file in metadata[dataset][mode]:
            if os.path.exists('./dataset/{}/'.format(dataset) + file + '.txt'):
                continue
            f = open('./dataset/{}/'.format(dataset) + file, 'r', encoding='utf-8')
            fw_event = open('./dataset/{}/'.format(dataset) + file + '.txt', 'a', encoding='utf-8')
            print('processing {} ...'.format(file))
            for line in tqdm(f):
                if 'com.bbn.tc.schema.avro.cdm18.Event' in line:
                    uuid = pattern_uuid.findall(line)[0]
                    record = pattern_record.findall(line)[0]
                    if record == 'Event':
                        # event_type = seq = thread_id = src = size = dst1 = dst2 = time = 'null'
                        event_type = src = dst1 = dst2 = time = 'null'
                        if len(pattern_type.findall(line)) > 0:
                            event_type = pattern_type.findall(line)[0]
                        # if len(pattern_seq.findall(line)) > 0:
                        #     seq = pattern_seq.findall(line)[0]
                        # if len(pattern_thread_id.findall(line)) > 0:
                        #     thread_id = pattern_thread_id.findall(line)[0]
                        if len(pattern_src.findall(line)) > 0:
                            src = pattern_src.findall(line)[0]
                        # if len(pattern_size.findall(line)) > 0:
                        #     size = pattern_size.findall(line)[0]
                        if len(pattern_dst1.findall(line)) > 0:
                            dst1 = pattern_dst1.findall(line)[0]
                        if len(pattern_dst2.findall(line)) > 0:
                            dst2 = pattern_dst2.findall(line)[0]
                        if len(pattern_time.findall(line)) > 0:
                            time = pattern_time.findall(line)[0]
                        # if 'READ' in event_type or 'RECV' in event_type or 'LOAD' in event_type:
                        # attr_event = str(uuid) + '\t' + str(record) + '\t' + str(event_type) + '\t' + str(
                        #     seq) + '\t' + str(thread_id) + '\t' + str(src) + '\t' + str(dst1) + '\t' + str(
                        #     dst2) + '\t' + str(size) + '\t' + str(time) + '\n'
                        attr_event = str(uuid) + '\t' + str(record) + '\t' + str(event_type) + '\t' + str(src) + '\t' + str(dst1) + '\t' + str(
                            dst2) + '\t' + str(time) + '\n'
                        fw_event.write(attr_event)
            fw_event.close()
#看恶意节点种类的
def malicious_type(dataset):
    malicious_entities = './groundtruth/{}.txt'.format(dataset)
    f = open(malicious_entities, 'r')
    malicious_entities = {}
    id_entity_map={}
    fw_malicious_type = open('./dataset/{}/'.format(dataset) + 'malicious_type.txt', 'a', encoding='utf-8')
    for l in f.readlines():
        entity = l.strip()
        if entity in malicious_entities:
            malicious_entities[entity] += 1  # 如果实体已存在，计数加1
        else:
            malicious_entities[entity] = 1  # 如果实体不存在，初始化为1
    # fw_malicious_type.write(malicious_entities)
    fw_malicious_entities = open('./dataset/{}/'.format(dataset) + 'malicious_entities.json', 'w', encoding='utf-8')
    json.dump(malicious_entities, fw_malicious_entities)
    fw_malicious_entities.close()
    if os.path.exists('./dataset/{}/id_entity_map.json'.format(dataset)):
        with open('./dataset/{}/id_entity_map.json'.format(dataset), 'r', encoding='utf-8') as f_id_entity_map:
            id_entity_map = json.load(f_id_entity_map)
    for i in malicious_entities:
        if malicious_entities[i]>1:
            print(i)
        type = id_entity_map[i]
        fw_malicious_type.write("{}\t{}\n".format(i,type))
    fw_malicious_type.close()

# 找出所有的节点对
def find_entity_pair(dataset):
    record_cnt_map = {}  # 记录event,src和dst这些的数字编号(uuid,cnt)
    entity_cnt = 0
    src_dst_deduplication_test = set()
    src_dst_deduplication_train = set()
    malicious_entities = './groundtruth/{}.txt'.format(dataset)
    f = open(malicious_entities, 'r')
    malicious_entities = set()
    for l in f.readlines():
        malicious_entities.add(l.lstrip().rstrip())
    if os.path.exists('./dataset/{}/id_entity_map.json'.format(dataset)):
        with open('./dataset/{}/id_entity_map.json'.format(dataset), 'r', encoding='utf-8') as f_id_entity_map:
            id_entity_map = json.load(f_id_entity_map)
    for file in metadata[dataset]['train']:
        entity_pairs = []
        path = './dataset/{}/'.format(dataset) + file + '.txt'
        f = open(path, 'r', encoding='utf-8')
        for l in f.readlines():
            split_line = l.split('\t')
            uuid, record, event_type,src, dst1, dst2,time = split_line
            # 训练的时候只用正常的数据进行训练,排除掉
            if src in malicious_entities and id_entity_map[src] != 'MemoryObject':
                continue
            if dst1 in malicious_entities and id_entity_map[src] != 'MemoryObject':
                continue
            if dst2 in malicious_entities and id_entity_map[src] != 'MemoryObject':
                continue
            attr_dict = ['Event', 'Subject', 'FileObject', 'NetFlowObject', 'MemoryObject']
            # 考虑加入unnamed
            if src in id_entity_map and dst1 in id_entity_map:
                if id_entity_map[src] not in attr_dict or id_entity_map[dst1] not in attr_dict:
                    continue
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
                if 'READ' in event_type or 'RECV' in event_type or 'LOAD' in event_type:
                    src_dst_pair = (dst1, src)
                    if src_dst_pair not in src_dst_deduplication_train:
                        src_dst_deduplication_train.add(src_dst_pair)
                        entity_pair = str(record_cnt_map[uuid]) + '\t' + str(record_cnt_map[src]) + '\t' + str(
                            record_cnt_map[dst1]) + '\t' + str(time)
                        entity_pairs.append(entity_pair)
                else:
                    src_dst_pair = (src, dst1)
                    if src_dst_pair not in src_dst_deduplication_train:
                        src_dst_deduplication_train.add(src_dst_pair)
                        entity_pair = str(record_cnt_map[uuid]) + '\t' + str(record_cnt_map[src]) + '\t' + str(
                            record_cnt_map[dst1]) + '\t' + str(time)
                        entity_pairs.append(entity_pair)
            if dst2 in id_entity_map:
                if id_entity_map[dst2] not in attr_dict:
                    continue
                if dst2 not in record_cnt_map:
                    record_cnt_map[dst2] = entity_cnt
                    entity_cnt += 1
                if 'READ' in event_type or 'RECV' in event_type or 'LOAD' in event_type:
                    src_dst_pair = (dst2, src)
                    if src_dst_pair not in src_dst_deduplication_train:
                        src_dst_deduplication_train.add(src_dst_pair)
                        entity_pair = str(record_cnt_map[uuid]) + '\t' + str(record_cnt_map[dst2]) + '\t' + str(
                            record_cnt_map[src]) + '\t' + str(time)
                        entity_pairs.append(entity_pair)
                else:
                    src_dst_pair = (src, dst2)
                    if src_dst_pair not in src_dst_deduplication_train:
                        src_dst_deduplication_train.add(src_dst_pair)
                        entity_pair = str(record_cnt_map[uuid]) + '\t' + str(record_cnt_map[src]) + '\t' + str(
                            record_cnt_map[dst2]) + '\t' + str(time)
                        entity_pairs.append(entity_pair)
        entity_pairs.sort(key=lambda l: l[4])
        with open('./dataset/{}/{}/'.format(dataset, 'train') + file+'.txt', "w") as fw_entity_pair:
            for pair in entity_pairs:
                fw_entity_pair.write(f"{pair}")
        fw_entity_pair.close()
        # finish the entity_pair.txt
    for file in metadata[dataset]['test']:
        entity_pairs = []
        path = './dataset/{}/'.format(dataset) + file + '.txt'
        f = open(path, 'r', encoding='utf-8')
        for l in f.readlines():
            split_line = l.split('\t')
            uuid, record, event_type, src, dst1, dst2,time = split_line
            attr_dict = ['Event', 'Subject', 'FileObject', 'NetFlowObject', 'MemoryObject']
            # 考虑加入unnamed
            if src in id_entity_map and dst1 in id_entity_map:
                if id_entity_map[src] not in attr_dict or id_entity_map[dst1] not in attr_dict:
                    continue
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
                if 'READ' in event_type or 'RECV' in event_type or 'LOAD' in event_type:
                    src_dst_pair = (dst1, src)
                    if src_dst_pair not in src_dst_deduplication_test:
                        src_dst_deduplication_test.add(src_dst_pair)
                        entity_pair = str(record_cnt_map[uuid]) + '\t' + str(record_cnt_map[src]) + '\t' + str(
                            record_cnt_map[dst1]) + '\t' + str(time)
                        entity_pairs.append(entity_pair)
                else:
                    src_dst_pair = (src, dst1)
                    if src_dst_pair not in src_dst_deduplication_test:
                        src_dst_deduplication_test.add(src_dst_pair)
                        entity_pair = str(record_cnt_map[uuid]) + '\t' + str(record_cnt_map[src]) + '\t' + str(
                            record_cnt_map[dst1]) + '\t' + str(time)
                        entity_pairs.append(entity_pair)
            if dst2 in id_entity_map:
                if id_entity_map[dst2] not in attr_dict:
                    continue
                if dst2 not in record_cnt_map:
                    record_cnt_map[dst2] = entity_cnt
                    entity_cnt += 1
                if 'READ' in event_type or 'RECV' in event_type or 'LOAD' in event_type:
                    src_dst_pair = (dst2, src)
                    if src_dst_pair not in src_dst_deduplication_test:
                        src_dst_deduplication_test.add(src_dst_pair)
                        entity_pair = str(record_cnt_map[uuid]) + '\t' + str(record_cnt_map[dst2]) + '\t' + str(
                            record_cnt_map[src]) + '\t' + str(time)
                        entity_pairs.append(entity_pair)
                else:
                    src_dst_pair = (src, dst2)
                    if src_dst_pair not in src_dst_deduplication_test:
                        src_dst_deduplication_test.add(src_dst_pair)
                        entity_pair = str(record_cnt_map[uuid]) + '\t' + str(record_cnt_map[src]) + '\t' + str(
                            record_cnt_map[dst2]) + '\t' + str(time)
                        entity_pairs.append(entity_pair)
        entity_pairs.sort(key=lambda l: l[4])
        with open('./dataset/{}/{}/'.format(dataset, 'test') + file+'.txt', "w") as fw_entity_pair:
            for pair in entity_pairs:
                fw_entity_pair.write(f"{pair}")
        fw_entity_pair.close()
        # finish the entity_pair.txt
    if len(record_cnt_map) != 0:
        fw_record_cnt_map = open('./dataset/{}/'.format(dataset) + 'record_cnt_map.json', 'w',
                                 encoding='utf-8')
        json.dump(record_cnt_map, fw_record_cnt_map)
        fw_record_cnt_map.close()
        # finish the record_map.txt
        cnt_record_map = {str(v): k for k, v in record_cnt_map.items()}
        # reverse
        fw_cnt_record_map = open('./dataset/{}/'.format(dataset) + 'cnt_record_map.json', 'w',
                                 encoding='utf-8')
        json.dump(cnt_record_map, fw_cnt_record_map)
        fw_cnt_record_map.close()
        # finish the count_record.txt
    return

def ip_to_binary_list(ip):
    if pd.isna(ip):
        return torch.zeros(128, dtype=torch.int32)
    ip_int = int(ipaddress.ip_address(ip))
    binary_list = [(ip_int >> (127 - i)) & 1 for i in range(128)]
    binary_tensor = torch.tensor(binary_list, dtype=torch.int32)
    return binary_tensor

def classify_port(port):
    if port < 1024:
        return 0  # 知名端口
    elif port < 49152:
        return 1  # 注册端口
    else:
        return 2  # 动态端口

# key_attr_dict = ['subject_type', 'path', 'remote_address', 'memory_address',
#                      'event_type']  # 对应subject,file,netflow,memory,event核心信息
# detail_attr_dict = [ 'cmdline', 'file_type', 'local_address', 'local_port', 'remote_port']
def one_hot_encode(df, column_name, max_dim=32):
    # 对列进行One-Hot编码
    encoder = OneHotEncoder(sparse_output=False)
    encoded = encoder.fit_transform(df[[column_name]])
    # 确保维度为max_dim，若不足128维，填充零；若超过128维，截断
    if encoded.shape[1] < max_dim:
        padding = np.zeros((encoded.shape[0], max_dim - encoded.shape[1]))
        encoded = np.hstack([encoded, padding])
    elif encoded.shape[1] > max_dim:
        encoded = encoded[:, :max_dim]
    print(f"{column_name} is ready")
    return encoded

def get_cnt(df, attr_type):
    if attr_type in ['remote_address', 'memory_address', 'local_address']:
        df[attr_type] = df[attr_type].apply(ip_to_binary_list)
        return df
    elif attr_type in ['local_port', 'remote_port']:
        df[attr_type] = df[attr_type].apply(classify_port)
        df[attr_type] = list(one_hot_encode(df, attr_type))
        return df
    elif attr_type == 'subject_type':
        df['subject_type'] = pd.Categorical(df['subject_type']).codes
        df[attr_type] = list(one_hot_encode(df, attr_type))
        return df
    elif attr_type == 'file_type':
        df['file_type'] = pd.Categorical(df['file_type']).codes
        df[attr_type] = list(one_hot_encode(df, attr_type))
        return df
    elif attr_type == 'event_type':
        df['event_type'] = pd.Categorical(df['event_type']).codes
        df[attr_type] = list(one_hot_encode(df, attr_type))
        return df
    else:
        raise NotImplementedError(f"This attribute type '{attr_type}' is not implemented yet.")

def get_embedding(df, attr_type):
    if attr_type == 'cmdline':
        # 将cmdline和path转换为特征向量
        vectorizer = TfidfVectorizer(max_features=128)  # 可以调整特征维度
        cmdline_features = vectorizer.fit_transform(df['cmdline'].fillna(''))  # 处理空值
        cmdline_features = cmdline_features.toarray()
        df['cmdline'] = list(cmdline_features)
        return df
    elif attr_type == 'path':
        vectorizer = TfidfVectorizer(max_features=128)  # 可以调整特征维度
        path_features = vectorizer.fit_transform(df['path'].fillna(''))  # 处理空值
        path_features = path_features.toarray()
        df['path'] = list(path_features)
        return df
    else:
        raise NotImplementedError("This type is not included")


def get_attrs(dataset, mode):
    # entity == subject
    uuid_to_node_attrs = {}
    uuid_to_edge_attrs = {}
    if os.path.exists('./dataset/{}/attr_subject.txt'.format(dataset)):
        with open('./dataset/{}/attr_subject.txt'.format(dataset), 'r', encoding='utf-8') as f_sub:
            df = pd.read_csv(f_sub,
                             sep='\t',
                             names=['uuid', 'record', 'subject_type', 'parent',
                                    'local_principal', 'cid', 'start_time',
                                    'unit_id', 'cmdline'])
            df = get_cnt(df, 'subject_type')
            df = get_embedding(df, 'cmdline')
            uuid_to_node_attrs.update(df.set_index('uuid').to_dict('index'))
    if os.path.exists('./dataset/{}/attr_file.txt'.format(dataset)):
        with open('./dataset/{}/attr_file.txt'.format(dataset), 'r', encoding='utf-8') as f_file:
            df = pd.read_csv(f_file,
                             sep='\t',
                             names=['uuid', 'record', 'file_type', 'epoch',
                                    'permission', 'path'])
            df = get_cnt(df, 'file_type')
            df = get_embedding(df, 'path')
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
            df = get_cnt(df, 'local_address')
            df = get_cnt(df, 'local_port')
            df = get_cnt(df, 'remote_address')
            df = get_cnt(df, 'remote_port')
            uuid_to_node_attrs.update(df.set_index('uuid').to_dict('index'))

    if os.path.exists('./dataset/{}/attr_memory.txt'.format(dataset)):
        with open('./dataset/{}/attr_memory.txt'.format(dataset), 'r', encoding='utf-8') as f_mem:
            df = pd.read_csv(f_mem,
                             sep='\t',
                             names=['uuid', 'record', 'epoch',
                                    'memory_address',
                                    'tgid',
                                    'size'])
            df = get_cnt(df, 'memory_address')
            uuid_to_node_attrs.update(df.set_index('uuid').to_dict('index'))

    # if os.path.exists('./dataset/{}/attr_unnamed.txt'.format(dataset)):
    #     with open('./dataset/{}/attr_unnamed.txt'.format(dataset), 'r', encoding='utf-8') as f_unnamed:
    #         df = pd.read_csv(f_unnamed,
    #                          sep='\t',
    #                          names=['uuid', 'record', 'epoch', 'pid', 'source_file_descriptor', 'sink_file_descriptor']
    #                          )
    #         uuid_to_node_attrs.update(df.set_index('uuid').to_dict('index'))

    if os.path.exists('./dataset/{}/attr_event.txt'.format(dataset)):
        with open('./dataset/{}/attr_event.txt'.format(dataset), 'r', encoding='utf-8') as f_event:
            df = pd.read_csv(f_event,
                             sep='\t',
                             names=['uuid', 'record', 'event_type', 'src', 'dst1', 'dst2','time'],
                             dtype={'size': 'float'},
                             usecols=['uuid', 'record', 'event_type','time']
                             )
            df = get_cnt(df, 'event_type')
            uuid_to_edge_attrs.update(df.set_index('uuid').to_dict('index'))
    with open('./dataset/{}/uuid_to_attrs.pkl'.format(dataset), 'wb') as f:
        pkl.dump((uuid_to_node_attrs, uuid_to_edge_attrs), f)

def get_maps(dataset):
    if os.path.exists('./dataset/{}/id_entity_map.json'.format(dataset)):
        with open('./dataset/{}/id_entity_map.json'.format(dataset), 'r', encoding='utf-8') as f_id_entity_map:
            print('loading id_entity_map')
            id_entity_map = json.load(f_id_entity_map)
    if os.path.exists('./dataset/{}/cnt_record_map.json'.format(dataset)):
        with open('./dataset/{}/cnt_record_map.json'.format(dataset), 'r',
                  encoding='utf-8') as f_cnt_record_map:
            print('loading cnt_record_map')
            cnt_record_map = json.load(f_cnt_record_map)
    return id_entity_map, cnt_record_map

def single_sub_g_construction(src_uuid, dst_uuid, event_uuid, uuid_to_node_attrs, uuid_to_edge_attrs):
    sub_g = nx.DiGraph()
    # key_attr_dict = ['subject_type', 'path', 'remote_address', 'memory_address',
    #                  'event_type']  # 对应subject,file,netflow,memory,event核心信息
    # detail_attr_dict = ['cmdline', 'file_type', 'local_address', 'local_port', 'remote_port']
    key_attr_dict = ['subject_type', 'path', 'remote_address','event_type']  # 对应subject,file,netflow,memory,event核心信息
    detail_attr_dict = ['cmdline', 'file_type']
    # subject_type 直接编号 remote_address 映射成0-2^32-1 memory_address 0-2^48-1 event_type 直接编号
    #  cmdline doc2vec file_type 直接编号 local_address映射成0-2^32-1 local_port、remote_port、ip_protocol
    cnt_node = 0
    src_node_cnt = 0
    dst_node_cnt = 0
    event_attr = uuid_to_edge_attrs[event_uuid]
    src_attr = uuid_to_node_attrs[src_uuid]
    dst_attr = uuid_to_node_attrs[dst_uuid]
    sub_g_nodes_list = []
    sub_g_edges_list = []

    # key node
    for attr_name, attr_value in src_attr.items():
        if attr_name in key_attr_dict:
            sub_g_nodes_list.append((cnt_node, {"type": attr_value}))
            # print('{}:{}'.format(attr_name,type(attr_value)))
            src_node_cnt = cnt_node
            cnt_node += 1
    for attr_name, attr_value in dst_attr.items():
        if attr_name in key_attr_dict:
            # print('{}:{}'.format(attr_name,type(attr_value)))
            sub_g_nodes_list.append((cnt_node, {"type": attr_value}))
            dst_node_cnt = cnt_node
            cnt_node += 1
    # key edge
    for attr_name, attr_value in event_attr.items():
        if attr_name in key_attr_dict:
            sub_g_edges_list.append((src_node_cnt, dst_node_cnt, {"type": attr_value}))
    # detail node
    for attr_name, attr_value in src_attr.items():
        if attr_name in detail_attr_dict:
            sub_g_nodes_list.append((cnt_node, {"type": attr_value}))
            sub_g_edges_list.append((src_node_cnt, cnt_node))
            cnt_node += 1
    for attr_name, attr_value in dst_attr.items():
        if attr_name in detail_attr_dict:
            sub_g_nodes_list.append((cnt_node, {"type": attr_value}))
            sub_g_edges_list.append((dst_node_cnt, cnt_node))
            cnt_node += 1
    sub_g.add_nodes_from(sub_g_nodes_list)
    sub_g.add_edges_from(sub_g_edges_list)
    return sub_g


def sub_g_embedding_aggregation(sub_g, max_dim=128):
    attr = nx.get_node_attributes(sub_g, 'type')
    processed_types = {}
    # 第二次遍历处理所有属性
    for attr_name, attr_value in attr.items():
        if isinstance(attr_value, (list, np.ndarray)):
            if len(attr_value) < max_dim:
                processed_types[attr_name] = np.array(attr_value, dtype=np.float32).tolist() + [0.0] * (
                        max_dim - len(attr_value))
            else:
                processed_types[attr_name] = attr_value
        else:
            processed_types[attr_name] = attr_value
    node_embeddings = []
    for attr_name, attr_value in processed_types.items():
        # if 'type' not in attr_name: # 只保留type测试
        #     continue
        if isinstance(attr_value, torch.Tensor):
            node_embeddings.append(attr_value)
        else:
            # print(attr_value, type(attr_value), attr_name)
            node_embeddings.append(torch.tensor(attr_value))
    # 将所有节点的特征相加
    sub_g_embedding = torch.stack(node_embeddings).sum(dim=0)
    return sub_g_embedding


def sub_g_embedding_construction(dataset, uuid_to_node_attrs, uuid_to_edge_attrs, id_entity_map, cnt_record_map, mode):
    cnt = 0
    malicious_cnt_list = []
    g_nodes_list = []
    malicious_entities = './groundtruth/{}.txt'.format(dataset)
    f = open(malicious_entities, 'r')
    malicious_entities = set()
    for l in f.readlines():
        malicious_entities.add(l.lstrip().rstrip())
    for file in metadata[dataset][mode]:
        path = './dataset/{}/{}/'.format(dataset,mode) + file + '.txt'
        g_nodes = []
        with open(path, 'r') as f:
            print('loading event_list {} for sub_g_construction'.format(file))
            for line in tqdm(f):
                event = line.strip().split('\t')
                entity_pair = {
                    'event': cnt_record_map[event[0]],
                    'src': cnt_record_map[event[1]],
                    'dst': cnt_record_map[event[2]],
                }
                # attr_dict = ['Event', 'Subject', 'FileObject', 'NetFlowObject', 'MemoryObject']
                # 这个判定条件移动到了find_entity_pair中
                # if id_entity_map[entity_pair['src']] in attr_dict and id_entity_map[entity_pair['dst']] in attr_dict:
                event_uuid = entity_pair['event']
                src_uuid = entity_pair['src']
                dst_uuid = entity_pair['dst']
                cnt += 1
                # if mode == 'test':
                #     if src_uuid in malicious_entities or dst_uuid in malicious_entities:
                #         malicious_cnt_list.append(cnt)
                if mode == 'test':
                    if src_uuid in malicious_entities and src_uuid:
                        malicious_cnt_list.append(cnt)
                sub_g = single_sub_g_construction(src_uuid, dst_uuid, event_uuid, uuid_to_node_attrs,
                                                  uuid_to_edge_attrs)
                sub_g_embedding = sub_g_embedding_aggregation(sub_g)
                g_nodes.append((cnt, {"attr": sub_g_embedding}))
        g_nodes_list.append(g_nodes)
    if mode == 'test':
        pkl.dump(malicious_cnt_list, open('./dataset/{}/test/malicious.pkl'.format(dataset), 'wb'))
    return g_nodes_list


# 若遇到节点出度入度过大，采样20条边添加，边以cnt形式表示，若node_list中发现edge_list中没有的cnt，说明该边没有被采样，所以舍弃该节点对

def graph_edge_construction(dataset, mode):
    g_edges_list=[]
    cnt = 0
    print('processing g_edges_list')
    for file in metadata[dataset][mode]:
        path = './dataset/{}/{}/'.format(dataset, mode) + file + '.txt'
        g_edges_set = set()
        with open(path, 'r', encoding='utf-8') as f:
            map_a = defaultdict(list)
            map_b = defaultdict(list)
            for line in f:
                cnt += 1
                event, src, dst, time = line.strip().split('\t')
                hash_dst = hash(dst)
                hash_src = hash(src)
                map_a[hash_dst].append(cnt)
                map_b[hash_src].append(cnt)
            for hash_dst in tqdm(map_a, total=len(map_a)):
                if hash_dst in map_b:
                    cnt_list_a = map_a[hash_dst]
                    cnt_list_b = map_b[hash_dst]
                    # 如果 cnt_list_a 或 cnt_list_b 的元素小于 100，则创建所有的边
                    if len(cnt_list_a) < 100 and len(cnt_list_b) < 100:
                        for event_src in cnt_list_a:
                            for event_dst in cnt_list_b:
                                if event_src != event_dst and (event_src, event_dst) not in g_edges_set:
                                    g_edges_set.add((event_src, event_dst))
                    else:
                        # 否则从 cnt_list_a 和 cnt_list_b 中各自随机采样 100 条数据
                        sampled_a = random.sample(cnt_list_a, min(100, len(cnt_list_a)))
                        sampled_b = random.sample(cnt_list_b, min(100, len(cnt_list_b)))

                        for event_src in sampled_a:
                            for event_dst in sampled_b:
                                if event_src != event_dst and (event_src, event_dst) not in g_edges_set:
                                    g_edges_set.add((event_src, event_dst))
            g_edges_list.append(list(g_edges_set))
    with open('./dataset/{}/{}/g_edges_list.pkl'.format(dataset, mode), 'wb') as f:
        pkl.dump(list(g_edges_list), f)


# 对每个节点对进行处理，构建子图
def graph_node_construction(dataset, mode):
    uuid_to_node_attrs = {}
    uuid_to_edge_attrs = {}
    g_nodes_list = []
    if os.path.exists('./dataset/{}/uuid_to_attrs.pkl'.format(dataset)):
        with open('./dataset/{}/uuid_to_attrs.pkl'.format(dataset), 'rb') as f:
            uuid_to_node_attrs, uuid_to_edge_attrs = pkl.load(f)
    else:
        raise NotImplementedError("There is not pkl file")
    id_entity_map, cnt_record_map = get_maps(dataset)
    g_nodes_list = sub_g_embedding_construction(dataset, uuid_to_node_attrs, uuid_to_edge_attrs, id_entity_map,
                                                cnt_record_map, mode)
    print("g_nodes_list is ready")
    with open('./dataset/{}/{}/g_nodes_list.pkl'.format(dataset, mode), 'wb') as f:
        pkl.dump(g_nodes_list, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Darpa TC E3 Parser')
    parser.add_argument("--dataset", type=str, default="trace")
    parser.add_argument("--mode", type=str, default="train")
    args = parser.parse_args()
    dataset = args.dataset
    mode = args.mode
    if dataset not in ['trace', 'theia', 'cadets']:
        raise NotImplementedError("This dataset is not included")
    # preprocess_entity_attr(dataset)
    #malicious_type(dataset)
    # preprocess(dataset) # 这里mode划分数据集
    # find_entity_pair(dataset) # 这里mode决定数据集中是否包含恶意节点
    # get_attrs(dataset,mode)
    # graph_node_construction(dataset, mode)
    graph_edge_construction(dataset, mode)

