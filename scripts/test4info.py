import json
import os
from tqdm import tqdm
def get_attrs(dataset,uuid_to_node_attrs,uuid_to_edge_attrs):
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
                uuid = node_attrs['uuid']
                uuid_to_node_attrs[uuid] = node_attrs

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
                uuid = node_attrs['uuid']
                uuid_to_node_attrs[uuid] = node_attrs
              

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
                uuid = node_attrs['uuid']
                uuid_to_node_attrs[uuid] = node_attrs
    
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
                uuid = node_attrs['uuid']
                uuid_to_node_attrs[uuid] = node_attrs
    
    if os.path.exists('./dataset/{}/attr_event.txt'.format(dataset)):
        with open('./dataset/{}/attr_event.txt'.format(dataset), 'r', encoding='utf-8') as f_event_sub:
            for attr in tqdm(f_event_sub.readlines()):
                attr_values = attr.strip().split('\t')
                # uuid	record	event_type	seq	thread_id	src	dst1	dst2	size	time
                edge_attrs = {
                    'uuid': attr_values[0],
                    'record': attr_values[1],
                    'event_type': attr_values[2],
                    'seq': attr_values[3],
                    'thread_id': attr_values[4],
                    'src': attr_values[5],
                    'dst1': attr_values[6],
                    'dst2': attr_values[7],
                    'size': attr_values[8],
                    'time': attr_values[9],
                }
                uuid = edge_attrs['uuid']
                uuid_to_edge_attrs[uuid] = edge_attrs
    return

def get_maps(dataset,id_entity_map,cnt_record_map):
    if os.path.exists('./dataset/{}/id_entity_map.json'.format(dataset)):
        with open('./dataset/{}/id_entity_map.json'.format(dataset), 'r', encoding='utf-8') as f_id_entity_map:
            print('loading id_entity_map')
            id_entity_map = json.load(f_id_entity_map)
    if os.path.exists('./dataset/{}/cnt_record_map.json'.format(dataset)):
        with open('./dataset/{}/cnt_record_map.json'.format(dataset), 'r', encoding='utf-8') as f_cnt_record_map:
            print('loading cnt_record_map')
            cnt_record_map = json.load(f_cnt_record_map)