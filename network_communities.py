#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:zq
@file: network_communities.py
@time: 2020/04/28
@Desc: 测试network找社区算法
"""

# !/usr/bin/python
# coding:utf-8
import sys
import networkx as nx
import time


#查找社区函数，k为完全子图的节点数
def find_community(graph,k):
    """
    
    :param graph: 图
    :param k: 完全子图包含的节点数
    :return: 社区列表
    """""
    return list(nx.algorithms.community.k_clique_communities(graph, k))


if __name__ == '__main__':

    wbNetwork = nx.erdos_renyi_graph(10000, 0.15)
    print("图的节点数：%d" % wbNetwork.number_of_nodes())
    print("图的边数：%d" % wbNetwork.number_of_edges())

    # 调用kclique社区算法
    for k in range(3, 6):
        print("############# k值: %d ################" % k)
        start_time = time.clock()
        rst_com = find_community(wbNetwork, k)
        end_time = time.clock()
        print("计算耗时(秒)：%.3f" % (end_time - start_time))
        print("生成的社区数：%d" % len(rst_com))