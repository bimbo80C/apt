import os
from tqdm import tqdm
import re
import json
import collections
from collections import Counter
import networkx as nx

def get_attrs():
    uuid_to_node_attrs = {}
    uuid_to_edge_attrs = {}
    if os.path.exists('./dataset/mini_attr_subject.txt'):
        with open('./dataset/mini_attr_subject.txt', 'r', encoding='utf-8') as f_attr_sub:
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

    if os.path.exists('./dataset/mini_attr_file.txt'):
        with open('./dataset/mini_attr_file.txt', 'r', encoding='utf-8') as fw_file_sub:
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
              

    if os.path.exists('./dataset/mini_attr_network.txt'):
        with open('./dataset/mini_attr_network.txt', 'r', encoding='utf-8') as fw_network_sub:
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
    
    if os.path.exists('./dataset/mini_attr_mem.txt'):
        with open('./dataset/mini_attr_mem.txt', 'r', encoding='utf-8') as f_mem_sub:
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
    
    if os.path.exists('./dataset/mini_attr_event.txt'):
        with open('./dataset/mini_attr_event.txt', 'r', encoding='utf-8') as f_event_sub:
            for attr in tqdm(f_event_sub.readlines()):
                attr_values = attr.strip().split('\t')
                # uuid	record	event_type	seq	thread_id	src	dst1	dst2	size	time
                edge_attrs = {
                    'record': attr_values[1],
                    'event_type': attr_values[2],
                    'seq': attr_values[3],
                    'thread_id': attr_values[4],
                    'size': attr_values[8],
                    'time': attr_values[9],
                }
                uuid = attr_values[0]
                uuid_to_edge_attrs[uuid] = edge_attrs
    return uuid_to_node_attrs, uuid_to_edge_attrs





def get_maps():
    id_entity_map = {}
    cnt_record_map = {}
    if os.path.exists('./dataset/mini_id_entity_map.json'):
        with open('./dataset/mini_id_entity_map.json', 'r', encoding='utf-8') as f_id_entity_map:
            print('loading id_entity_map')
            id_entity_map = json.load(f_id_entity_map)
    if os.path.exists('./dataset/mini_cnt_record_map.json'):
        with open('./dataset/mini_cnt_record_map.json', 'r', encoding='utf-8') as f_cnt_record_map:
            print('loading cnt_record_map')
            cnt_record_map = json.load(f_cnt_record_map)
    return id_entity_map, cnt_record_map 




def graph_node_construction(graph_list,uuid_to_node_attrs,uuid_to_edge_attrs,id_entity_map,cnt_record_map):
    if os.path.exists('./dataset/mini_entity_pair.txt'):
        with open('./dataset/mini_entity_pair.txt', 'r', encoding='utf-8') as f:
            print('loading event_list')
            for line in f.readlines():
                event = line.strip().split('\t')
                entity_pair ={
                    'event': cnt_record_map[event[0]],
                    'src': cnt_record_map[event[1]],
                    'dst': cnt_record_map[event[2]],
                }
                attr_dict = ['Event','Subject','FileObject','NetFlowObject','MemoryObject']
                if id_entity_map[cnt_record_map[event[1]]] in attr_dict and id_entity_map[cnt_record_map[event[2]]] in attr_dict:
                    event_uuid = entity_pair['event']
                    src_uuid = entity_pair['src']
                    dst_uuid = entity_pair['dst']
                    event_attr = uuid_to_edge_attrs[event_uuid]
                    src_attr = uuid_to_node_attrs[src_uuid]
                    dst_attr = uuid_to_node_attrs[dst_uuid]
                    sub_G = nx.DiGraph()
                    sub_nodes_dict = {}
                    sub_edges_dict = {}
                    # for attr_name,attr_value in src_attr.items():
                    #     sub_nodes_dict[attr_name+str(event[1])] = attr_name + str(':') + attr_value
                    #     if attr_name != 'uuid':
                    #         sub_edges_dict[(str('uuid:')+src_uuid, attr_name + str(':') + attr_value)] = None
                        
                    # for attr_name, attr_value in dst_attr.items():
                    #     sub_nodes_dict[attr_name+str(event[2])] = attr_name + str(':') + attr_value
                    #     if attr_name != 'uuid':
                    #         sub_edges_dict[(str('uuid:')+dst_uuid, attr_name + str(':') + attr_value)] = None
                    # sub_edges_dict[(str('uuid:')+src_uuid,str('uuid:')+dst_uuid)] = event_attr
                    # sub_G.add_nodes_from(sub_nodes_dict.values())
                    cnt_node = 0
                    for attr_name,attr_value in src_attr.items():
                        sub_nodes_dict[cnt_node] = attr_name + str(':') + attr_value
                        cnt_node +=1
                        if attr_name != 'uuid':
                            sub_edges_dict[(str('uuid:')+src_uuid, attr_name + str(':') + attr_value)] = None
                        
                    for attr_name, attr_value in dst_attr.items():
                        sub_nodes_dict[cnt_node] = attr_name + str(':') + attr_value
                        cnt_node +=1
                        if attr_name != 'uuid':
                            sub_edges_dict[(str('uuid:')+dst_uuid, attr_name + str(':') + attr_value)] = None
                    sub_edges_dict[(str('uuid:')+src_uuid,str('uuid:')+dst_uuid)] = event_attr
                    sub_G.add_nodes_from(sub_nodes_dict.values())


                    # for node_data in sub_nodes_dict.values():
                    #     sub_G.add_node(node_data)
                    
                    # print(sub_nodes_dict)
                    # print(sub_G.nodes())
                    # assert 1==0
                    sub_G.add_edges_from(sub_edges_dict.keys())

                    # print(sub_G.edges())
                    num_nodes = sub_G.number_of_nodes()
                    num_edges = sub_G.number_of_edges()
                    # sub_G = nx.DiGraph()
                    graph_list.append(sub_G)
                    for node in graph_list[0].nodes():
                        print(node)
    return


graph_list = []
id_entity_map = {}
cnt_record_map = {}
uuid_to_node_attrs = {}
uuid_to_edge_attrs = {}

uuid_to_node_attrs,uuid_to_edge_attrs = get_attrs()
id_entity_map, cnt_record_map = get_maps()
graph_node_construction(graph_list,uuid_to_node_attrs,uuid_to_edge_attrs,id_entity_map,cnt_record_map)