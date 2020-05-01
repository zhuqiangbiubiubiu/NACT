#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:zq
@file: find_keymans_in_skitter.py
@time: 2020/04/30
@Desc: find keymans in skitter data set

"""
from find_keymans import *
import pandas as pd

def load_skitter_data(filepath):
    """
    load as-skitter data
    :param filepath: file store page
    :return:vertexs list ,edges list
    """
    ids = []
    edges = []


    with open(filepath, 'r') as f:
        lines = f.readlines()
        series = pd.Series(lines)
        series = series.str.rstrip('\n')
        series = series[5:]
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


if __name__ == '__main__':
    ids, edges,total_nodes,total_edges = load_skitter_data(r"./data/as-skitter.txt")
    G = nx.Graph()
    for i in range(1696415):
        G.add_node(str(i))
    for edge in edges:
        try:
            G.add_edge(edge[0], edge[1])
        except:
            continue
    Undirected_G = G
    degree_dict = get_global_degree(Undirected_G)
    communities = find_communities(Undirected_G)
    for i in range(1,4):
        numKeyMans = int(total_nodes * i / 1000)
        print("Prepare to select {} ‰ / {} keymans to test.".format(i, numKeyMans))

        KeyMans = []
        for com_id in set(communities.values()):
            list_nodes = [nodes for nodes in communities.keys()
                          if communities[nodes] == com_id]
            numKeyMans_per_com = int(len(list_nodes) / total_nodes * numKeyMans) + 1
            keymans = find_keymans_in_one_community(degree_dict, list_nodes, numKeyMans_per_com)
            KeyMans.extend(keymans)
        count = 0
        calculate_coverage(total_nodes, KeyMans, Undirected_G)