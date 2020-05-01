#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:zq
@file: generate_graph.py
@time: 2020/04/29
@Desc: 生成社交网络图
"""
import networkx as nx
import pandas as pd
import os
import os.path as osp
import numpy as np
import community
import matplotlib.pyplot as plt




def data_loader(filepath):
    ids = []
    edges = []
    fnames = [fn for fn in os.listdir('./data/gplus') if 'edges' in fn]
    for i, fn in enumerate(fnames):
        print('{}/{}: {}'.format(i + 1, len(fnames), fn))
        with open(osp.join('./data/gplus', fn), 'r') as f:
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
            break

    ids = list(set(list(ids)))
    print('Number of nodes:', len(ids))
    print('Number of edges:', len(edges))
    return ids,edges

def undirect_graph(ids,edges):
    """
    构造无向图
    :param ids:
    :param edges:
    :return:
    """
    ug = nx.Graph()
    ug.add_nodes_from(ids)
    ug.add_edges_from(edges)
    return ug

def find_communities(G):
    """
    找寻社区
    :param G:
    :return:
    """
    # G = nx.erdos_renyi_graph(30, 0.05)

    # first compute the best partition
    print("Start Louvain Algorithm ")
    partition = community.best_partition(G)
    # print(partition)

    # # drawing
    # size = float(len(set(partition.values())))
    # pos = nx.spring_layout(G)
    # count = 0.

    for com in set(partition.values()):
        # count = count + 1.
        list_nodes = [nodes for nodes in partition.keys()
                      if partition[nodes] == com]
        print("Nodes in Community {}:{}".format(com,len(list_nodes)))
        # nx.draw_networkx_nodes(G, pos, list_nodes, node_size=20,
        #                        node_color=str(count / size))
        break
    # print("Communities Count {}".format(count))
    # nx.draw_networkx_edges(G, pos, alpha=0.5)
    # plt.savefig("./communities.png")
    # # plt.show()
    print("Finish Louvain Algorithm ")
    return partition


def post_process(communities):
    """

    :param communities:
    :return:
    """

if __name__ == '__main__':
    ids,edges =data_loader("")
    G = undirect_graph(ids,edges)
    # pos = nx.spring_layout(G)
    # nx.draw_networkx_edges(G, pos, alpha=0.5,node_size=20)
    # plt.show()
    # plt.savefig("./raw_graph.png")
    find_communities(G)

