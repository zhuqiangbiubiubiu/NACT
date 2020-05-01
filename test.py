#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:zq
@file: test.py
@time: 2020/04/27
@Desc:
"""
import numpy as np
import sys
import pandas as pd
import time
from functools import wraps

filepath = r"D:\DIDI\chengdu\gps_20161001"

#计算时间消耗的装饰器函数
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
def data_loader(filepath,chunkSize=100000):
    """
    分块读取较大的csv文件，并对其中的坐标点和时间点进行映射处理
    :param filepath: 文件名
    :param chunkSize: 读取块大小
    :return:
    """
    colsname = ['CarId', 'OrderId', 'Timestamp', 'Lng', 'Lat']
    reader = pd.read_csv(filepath,names=colsname,iterator=True,engine='python')
    loop = True
    chunks = []
    while loop:
        try:

            chunk = reader.get_chunk(chunkSize)
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print('Iteration is stopped.')
    data = pd.concat(chunks,ignore_index=True)
    return data

# @get_func_exetime
# def data_loader2(filepath):
#     with open("test.txt") as f:
#         for line in f:



if __name__ == '__main__':
    data = data_loader(filepath)
    print(f"CarID Num:{len(data['CarId'].unique())}")
    print(f"OrderID Num:{len(data['OrderId'].unique())}")

