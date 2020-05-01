#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:zq
@file: find_keymans.py
@time: 2020/04/29
@Desc: find key mans in social network
"""
import community
import numpy as np
import pandas as pd
import os
import os.path as osp
import time
from functools import wraps
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import json

import sys
import heapq
# matplotlib.use("Agg")

class TopKHeap(object):
    def __init__(self, k):
        self.k = k
        self.data = []

    def push(self, elem):
        if len(self.data) < self.k:
            heapq.heappush(self.data, elem)
        else:
            topk_small = self.data[0]
            if elem[1] > topk_small[1]:
                heapq.heapreplace(self.data, elem)
    def topk(self):
        return [x for x in [heapq.heappop(self.data) for x in range(len(self.data))]]




#计算时间函数
def get_func_exetime(func):
    @wraps(func)
    def func_cost_time(*args,**kwargs):
        t0 = time.time()
        result = func(*args,**kwargs)
        t1 = time.time()
        print('Total time running %s : %s seconds'%(func.__name__,str(round(t1 - t0,3))))
        return result
    return func_cost_time

@get_func_exetime
def data_loader(filedir='./data/gplus'):
    """
    load social network data
    :param filedir: file dir
    :return: vertexs list,edges list
    """
    ids = []
    edges = []
    fnames = [fn for fn in os.listdir(filedir) if 'edges' in fn]
    for i, fn in enumerate(fnames):
        print('{}/{}: {}'.format(i + 1, len(fnames), fn))
        with open(osp.join(filedir, fn), 'r') as f:
            lines = f.readlines()
            if len(lines) == 0:
                continue
            series = pd.Series(lines)
            series = series.str.rstrip('\n')
            series = series.str.split()
            edges_ = list(series)

            ids_ = np.array(edges_).reshape([-1])
            ids_ = list(set(list(ids_)))

            ids += ids_
            edges += edges_
            # break
    ids = list(set(list(ids)))
    print('Number of nodes:', len(ids))
    print('Number of edges:', len(edges))
    return ids, edges, len(ids), len(edges)

@get_func_exetime
def building_social_network(vertexs,edges):
    """
    building graoh structure social network
    :param vertexs: vertex list
    :param edges: edge list
    :return: graoh structure social network
    """
    Undirected_G = nx.Graph()
    Undirected_G.add_nodes_from(vertexs)
    Undirected_G.add_edges_from(edges)
    return Undirected_G

@get_func_exetime
def find_communities(G,save_file='./communities.json',if_draw=False):
    """
    find communities in undirected graph
    :param G: Undirected Graph
    :param save_file: save temp communities
    :param if_draw: weather to display the results
    :return: Communities in Undirected Graph
    """
    print("Start Louvain Algorithm... ")
    partition = community.best_partition(G)
    if if_draw:
        # drawing
        size = float(len(set(partition.values())))
        pos = nx.spring_layout(G)
        count = 0.

        for com in set(partition.values()):
            # count = count + 1.
            list_nodes = [nodes for nodes in partition.keys()
                          if partition[nodes] == com]
            # print("Nodes in Community {}:{}".format(com, len(list_nodes)))
            nx.draw_networkx_nodes(G, pos, list_nodes, node_size=20,
                                   node_color=str(count / size))
            break
        print("Communities Count {}".format(count))
        nx.draw_networkx_edges(G, pos, alpha=0.5)
        # plt.savefig("./communities.png")
        plt.show()

    json_str = json.dumps(partition, indent=4)
    with open(save_file, 'w') as json_file:
        json_file.write(json_str)

    print("Finish Louvain Algorithm... ")
    return partition


def get_global_degree(G):
    """
    获取每个节点的度
    :param G: Undirected Graph
    :return:  图中所有节点的度
    """
    degree_tuple_list = G.degree()
    degree_dict = dict()
    for node_degree_pair in degree_tuple_list:
        node_id, node_degree = node_degree_pair
        degree_dict[node_id] = node_degree

    return degree_dict


def find_keymans_in_one_community(degree_dict,Community_nodes,num_KeyMan):
    """

    :param degree_dict: degree for per node
    :param Community_nodes: memeber of current community
    :param num_KeyMan: number of key man
    :return: KeyMans in current commnunity
    """
    TopDegreeMans = TopKHeap(num_KeyMan)
    for node in Community_nodes:
        # print(TopDegreeMans.data)
        TopDegreeMans.push((node,degree_dict[node]))

    KeyMans_list = TopDegreeMans.topk()
    del TopDegreeMans
    KeyMans_list = [nodes_d[0] for nodes_d in KeyMans_list]
    # print(len(KeyMans_list))
    return KeyMans_list

def get_adj_node(keyman,G):
    """
    get current keyman's adj nodes
    :param keyman: keyman
    :param G: Undirected Graph
    :return: adj node list
    """
    return [n for n in G.neighbors(keyman)]

def calculate_coverage(total_nodes,keymans,G):
    """
    calculate coverage for all keymans
    :param total_nodes: number of nodes
    :param keymans: all keymans
    :param G
    :return:coverage info
    """
    # print(keymans)
    coverage_nodes = []
    for keyman in keymans:
        coverage_nodes += get_adj_node(keyman,G)
    coverage_num = len(set(coverage_nodes))

    print_str = "Test {} keymans can represent {} man. Coverage rate = {} % "\
        .format(len(keymans),coverage_num,round(coverage_num / total_nodes * 100,2))
    print(print_str)


if __name__ == '__main__':
    vertexs, edges, total_nodes, total_edges = data_loader()
    Undirected_G = building_social_network(vertexs,edges)
    degree_dict = get_global_degree(Undirected_G)
    communities = find_communities(Undirected_G)
    for i in range(1,4):
        numKeyMans = int(total_nodes * i / 1000)
        print("Prepare to select {} ‰ / {} keymans to test.".format(i,numKeyMans))
        KeyMans = []
        for com_id in set(communities.values()):
            list_nodes = [nodes for nodes in communities.keys()
                          if communities[nodes] == com_id]
            numKeyMans_per_com = int(len(list_nodes) / total_nodes * numKeyMans) + 1
            keymans = find_keymans_in_one_community(degree_dict,list_nodes,numKeyMans_per_com)
            KeyMans.extend(keymans)
        count = 0
        calculate_coverage(total_nodes,KeyMans,Undirected_G)








    # print(degree_dict)



