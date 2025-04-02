import argparse
import json
import os

import xxhash
import tqdm
import logging
import networkx as nx
from tqdm import tqdm
import time
import datetime


valid_node_type = ['file', 'process_memory', 'task', 'mmaped_file', 'path', 'socket', 'address', 'link']
CONSOLE_ARGUMENTS = None


def hashgen(l):
    hasher = xxhash.xxh64()
    for e in l:
        hasher.update(e)
    return hasher.intdigest()
def parse_nodes(json_string, node_map):
    json_object = None
    json_object = json.loads(json_string)
    if "activity" in json_object:
        activity = json_object["activity"]
        for uid in activity:
            if not uid in node_map:  # only parse unseen nodes
                if "prov:type"  in activity[uid]:
                    # a node must have a type.
                    # record this issue if logging is turned on
                    node_map[uid] = activity[uid]["prov:type"]
                    # 事件存了"prov:type":"socket"

    if "entity" in json_object:
        entity = json_object["entity"]
        for uid in entity:
            if not uid in node_map:
                if "prov:type" in entity[uid]:
                    node_map[uid] = entity[uid]["prov:type"]


def parse_all_nodes(filename, node_map):
    description = '\x1b[6;30;42m[STATUS]\x1b[0m Parsing nodes in CamFlow data from {}'.format(filename)
    pb = tqdm(desc=description, mininterval=1.0, unit=" recs")
    with open(filename, 'r') as f:
        # each line in CamFlow data could contain multiple
        # provenance nodes, we call @parse_nodes routine.
        for line in f:
            pb.update()  # for progress tracking
            parse_nodes(line, node_map)
    f.close()
    pb.close()


def parse_all_edges(inputfile, outputfile, node_map):
    total_edges = 0
    output = open(outputfile, "w+")
    description = '\x1b[6;30;42m[STATUS]\x1b[0m Parsing edges in CamFlow data from {}'.format(inputfile)
    pb = tqdm(desc=description, mininterval=1.0, unit=" recs")
    with open(inputfile, 'r') as f:
        for line in f:
            pb.update()
            json_object = json.loads(line)
            if "used" in json_object:
                used = json_object["used"]
                for uid in used:
                    edgetype = "used"
                    timestamp = used[uid]["cf:id"]
                    if "prov:entity" not in used[uid]:
                        continue
                    if "prov:activity" not in used[uid]:
                        continue
                    srcUUID = used[uid]["prov:entity"]
                    dstUUID = used[uid]["prov:activity"]
                    if srcUUID not in node_map:
                        continue
                    else:
                        srcVal = node_map[srcUUID]
                    if dstUUID not in node_map:
                        continue
                    else:
                        dstVal = node_map[dstUUID]
                    total_edges += 1
                    output.write("{}\t{}\t{}\t{}:{}:{}:{}\n".format(total_edges,hashgen([srcUUID]), hashgen([dstUUID]), srcVal, dstVal, edgetype, timestamp))

            if "wasGeneratedBy" in json_object:
                wasGeneratedBy = json_object["wasGeneratedBy"]
                for uid in wasGeneratedBy:
                    if "prov:type" not in wasGeneratedBy[uid]:
                        continue
                    else:
                        edgetype = "wasGeneratedBy"
                    if "cf:id" not in wasGeneratedBy[uid]:
                        continue
                    else:
                        timestamp = wasGeneratedBy[uid]["cf:id"]
                    if "prov:entity" not in wasGeneratedBy[uid]:
                        continue
                    if "prov:activity" not in wasGeneratedBy[uid]:
                        continue
                    srcUUID = wasGeneratedBy[uid]["prov:activity"]
                    dstUUID = wasGeneratedBy[uid]["prov:entity"]
                    if srcUUID not in node_map:
                        continue
                    else:
                        srcVal = node_map[srcUUID]
                    if dstUUID not in node_map:
                        continue
                    else:
                        dstVal = node_map[dstUUID]
                    total_edges += 1
                    output.write("{}\t{}\t{}\t{}:{}:{}:{}\n".format(total_edges,hashgen([srcUUID]), hashgen([dstUUID]), srcVal, dstVal, edgetype, timestamp))

            if "wasInformedBy" in json_object:
                wasInformedBy = json_object["wasInformedBy"]
                for uid in wasInformedBy:
                    if "prov:type" not in wasInformedBy[uid]:
                        continue
                    else:
                        edgetype = "wasInformedBy"
                    if "cf:id" not in wasInformedBy[uid]:
                        continue
                    else:
                        timestamp = wasInformedBy[uid]["cf:id"]
                    if "prov:informant" not in wasInformedBy[uid]:
                        continue
                    if "prov:informed" not in wasInformedBy[uid]:
                        continue
                    srcUUID = wasInformedBy[uid]["prov:informant"]
                    dstUUID = wasInformedBy[uid]["prov:informed"]
                    if srcUUID not in node_map:
                        continue
                    else:
                        srcVal = node_map[srcUUID]
                    if dstUUID not in node_map:
                        continue
                    else:
                        dstVal = node_map[dstUUID]
                    total_edges += 1
                    output.write("{}\t{}\t{}\t{}:{}:{}:{}\n".format(total_edges,hashgen([srcUUID]), hashgen([dstUUID]), srcVal, dstVal, edgetype, timestamp))

            if "wasDerivedFrom" in json_object:
                wasDerivedFrom = json_object["wasDerivedFrom"]
                for uid in wasDerivedFrom:
                    if "prov:type" not in wasDerivedFrom[uid]:
                        continue
                    else:
                        edgetype = "wasDerivedFrom"
                    if "cf:id" not in wasDerivedFrom[uid]:
                        continue
                    else:
                        timestamp = wasDerivedFrom[uid]["cf:id"]
                    if "prov:usedEntity" not in wasDerivedFrom[uid]:
                        continue
                    if "prov:generatedEntity" not in wasDerivedFrom[uid]:
                        continue
                    srcUUID = wasDerivedFrom[uid]["prov:usedEntity"]
                    dstUUID = wasDerivedFrom[uid]["prov:generatedEntity"]
                    if srcUUID not in node_map:
                        continue
                    else:
                        srcVal = node_map[srcUUID]
                    if dstUUID not in node_map:
                        continue
                    else:
                        dstVal = node_map[dstUUID]
                    total_edges += 1
                    output.write("{}\t{}\t{}\t{}:{}:{}:{}\n".format(total_edges,hashgen([srcUUID]), hashgen([dstUUID]), srcVal, dstVal, edgetype, timestamp))

            if "wasAssociatedWith" in json_object:
                wasAssociatedWith = json_object["wasAssociatedWith"]
                for uid in wasAssociatedWith:
                    if "prov:type" not in wasAssociatedWith[uid]:
                        continue
                    else:
                        edgetype = "wasAssociatedWith"
                    if "cf:id" not in wasAssociatedWith[uid]:
                        continue
                    else:
                        timestamp = wasAssociatedWith[uid]["cf:id"]
                    if "prov:agent" not in wasAssociatedWith[uid]:
                        continue
                    if "prov:activity" not in wasAssociatedWith[uid]:
                        continue
                    srcUUID = wasAssociatedWith[uid]["prov:agent"]
                    dstUUID = wasAssociatedWith[uid]["prov:activity"]
                    if srcUUID not in node_map:
                        continue
                    else:
                        srcVal = node_map[srcUUID]
                    if dstUUID not in node_map:
                        continue
                    else:
                        dstVal = node_map[dstUUID]
                    total_edges += 1
                    output.write("{}\t{}\t{}:{}:{}:{}\n".format(hashgen([srcUUID]), hashgen([dstUUID]), srcVal, dstVal,edgetype, timestamp))
    f.close()
    output.close()
    pb.close()
    return total_edges

# def read_single_graph(file_name):
#     graph = []
#     edge_cnt = 0
#     with open(file_name, 'r') as f:
#         for line in f:
#             try:
#                 edge = line.strip().split("\t")
#                 new_edge = [edge[0], edge[1]]
#                 attributes = edge[2].strip().split(":")
#                 source_node_type = attributes[0]
#                 destination_node_type = attributes[1]
#                 edge_type = attributes[2]
#                 edge_order = attributes[3]

#                 new_edge.append(source_node_type)
#                 new_edge.append(destination_node_type)
#                 new_edge.append(edge_type)
#                 new_edge.append(edge_order)
#                 graph.append(new_edge)
#                 edge_cnt += 1
#             except:
#                 print("{}".format(line))
#     #new_edge 0 1 2 3分别存放src_attr dst_attr edge_type edge_order
#     f.close()
#     graph.sort(key=lambda e: e[5])
#     return graph


# def process_graph(name):
#     graph = read_single_graph(name)
#     result_graph = nx.DiGraph()
#     cnt = 0
#     for num, edge in enumerate(graph):
#         cnt += 1
#         src, dst, src_type, dst_type, edge_type = edge[:5]
#         if src_type in valid_node_type and dst_type in valid_node_type:
#             if not result_graph.has_node(src):
#                 result_graph.add_node(src, type=src_type)
#             if not result_graph.has_node(dst):
#                 result_graph.add_node(dst, type=dst_type)
#             if not result_graph.has_edge(src, dst):
#                 result_graph.add_edge(src, dst, type=edge_type)
#     return cnt, result_graph


# node_type_list = []
# edge_type_list = []
# node_type_dict = {}
# edge_type_dict = {}


# def format_graph(g, name):
#     new_g = nx.DiGraph()
#     node_map = {}
#     node_cnt = 0
#     for n in g.nodes:
#         node_map[n] = node_cnt
#         new_g.add_node(node_cnt, type=g.nodes[n]['type'])
#         node_cnt += 1
#     for e in g.edges:
#         new_g.add_edge(node_map[e[0]], node_map[e[1]], type=g.edges[e]['type'])
#     for n in new_g.nodes:
#         node_type = new_g.nodes[n]['type']
#         if not node_type in node_type_dict:
#             node_type_list.append(node_type)
#             node_type_dict[node_type] = 1
#         else:
#             node_type_dict[node_type] += 1
#     for e in new_g.edges:
#         edge_type = new_g.edges[e]['type']
#         if not edge_type in edge_type_dict:
#             edge_type_list.append(edge_type)
#             edge_type_dict[edge_type] = 1
#         else:
#             edge_type_dict[edge_type] += 1
#     for n in new_g.nodes:
#         new_g.nodes[n]['type'] = node_type_list.index(new_g.nodes[n]['type'])
#     for e in new_g.edges:
#         new_g.edges[e]['type'] = edge_type_list.index(new_g.edges[e]['type'])
#     with open('{}.json'.format(name), 'w', encoding='utf-8') as f:
#         json.dump(nx.node_link_data(new_g), f)

def find_entity_pair(input):
    cnt = 25
    for i in tqdm(range(cnt)):
        file_name = '{}{}.log'.format(input, i + 1)
        with open(file_name, 'r') as f:
            for line in f:
                edge = line.strip().split("\t")
                # new_edge = [edge[0], edge[1]]
                # attributes = edge[2].strip().split(":")
                # source_node_type = attributes[0]
                # destination_node_type = attributes[1]
                # edge_type = attributes[2]
                # edge_order = attributes[3]

                # new_edge.append(source_node_type)
                # new_edge.append(destination_node_type)
                # new_edge.append(edge_type)
                # new_edge.append(edge_order)
                # graph.append(new_edge)
                # edge_cnt += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Unicorn Wget Parser')
    parser.add_argument("--dataset", type=str, default="wget")
    args = parser.parse_args()
    dataset = args.dataset


    args.input = './dataset/wget/origin_log/'
    args.output = './dataset/wget/processed/'
    args.final_output = './dataset/wget/final/'
    if not os.path.exists(args.input):
        os.mkdir(args.input)
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    if not os.path.exists(args.final_output):
        os.mkdir(args.final_output)
    CONSOLE_ARGUMENTS = args
    cnt = 0
    # 从日志当中预处理出需要的属性
    for fname in os.listdir(args.input):
        cnt += 1
        node_map = dict()
        parse_all_nodes(args.input + '/{}'.format(fname), node_map)
        parse_all_edges(args.input + '/{}'.format(fname), args.output + '/{}.log'.format(cnt), node_map)
    # 预处理后的建图
    input = args.output
    base = args.final_output
    line_cnt = 0
    # for i in tqdm(range(cnt)):
    #     single_cnt, result_graph = process_graph('{}{}.log'.format(input, i + 1))
    #     format_graph(result_graph, '{}{}'.format(base, i))
    #     line_cnt += single_cnt

    # print(line_cnt // 150)
    # print(len(node_type_list))
    # print(node_type_dict)
    # print(len(edge_type_list))
    # print(edge_type_dict)

