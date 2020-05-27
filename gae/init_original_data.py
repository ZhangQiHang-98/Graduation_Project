# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 16:28:40 2020
生成初始拓扑，拓扑节点满足通信范围与最大度两个条件
拓扑以大规模无标度无线传感器为例子
@author: Administrator
"""

import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import math
import operator
import utils
import os


# 参数设置
nodeSize = 100
footPrint = 500
communicationRange = 200
maxDegree = 25
m = 2   # 节点每次联接边的个数
nodeList = []


# 定义节点类,linked为相连的节点，neighbors为邻居节点
class Node:

    comRange = communicationRange
    
    def __init__(self, number, location):
        self.number = number
        self.location = {'x': location[0], 'y': location[1]}
        self.neighbors = []
        self.degree = 0
        self.linked = []
        self.visited = 0


# 计算欧式距离
def CalDistance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        

# 初始化节点，同时返回最中心节点 
def InitNodes():
    centerNode = 0
    minDistance = math.inf
    # 随机生成节点，传入节点编号与位置
    for i in range(nodeSize):
        x_pos = random.uniform(0, footPrint)
        y_pos = random.uniform(0, footPrint)
        nodeList.append(Node(i, [x_pos, y_pos]))
        # 记录离中心的距离，返回最中心节点来进行传播
        if CalDistance(x_pos, y_pos, footPrint / 2, footPrint / 2) < minDistance:
            minDistance = CalDistance(x_pos, y_pos, footPrint / 2, footPrint / 2)
            centerNode = i
    # 为每个节点添加候选邻居节点
    for i in range(nodeSize):
        nodeList[i].neighbors = FindNerbors(i)
    return nodeList[centerNode]


# 节点候选邻居函数
def FindNerbors(no):
    x1 = nodeList[no].location['x']
    y1 = nodeList[no].location['y']
    neibors = []
    for i in range(nodeSize):
        if i == no:
            continue
        x2 = nodeList[i].location['x']
        y2 = nodeList[i].location['y']
        if CalDistance(x1, y1, x2, y2) < communicationRange:
            neibors.append({'no': i, 'distance': CalDistance(x1, y1, x2, y2)})
    # 按距离排序传回，方便传播函数
    neibors = sorted(neibors, key=operator.itemgetter('distance'))
    return neibors


# 轮盘选择函数
def RouletteChosen(unZeroList):
    sumDegree = 0
    accuProb = 0
    # 轮盘选择列表
    rouletteList = []
    chosenList = []
    # 得到公式分母
    for i in unZeroList:
        sumDegree += nodeList[i].degree
        # print("节点",nodeList[i].number,"度",nodeList[i].degree)
    probList = [nodeList[x].degree / sumDegree for x in unZeroList]
    # 轮盘选择法
    for prob in probList:
        accuProb += prob
        rouletteList.append(accuProb)
    for i in range(m):
        number = random.random()
        for prob in rouletteList:
            if number <= prob:
                chosenList.append(rouletteList.index(prob))
                break
    return chosenList


# 建立连接
def BulidLink(neibors):
    # 求邻居节点中的0度邻居和有度邻居
    zeroList = [x['no'] for x in neibors if nodeList[x['no']].degree == 0]
    allList = [x['no'] for x in neibors]
    unZeroList = list(set(allList) - set(zeroList))
    # 打乱零度邻居，来实现零度邻居等概率被选中,自己理解的实现方式
    random.shuffle(zeroList)
    if m == len(unZeroList):
        # print("直接返回度高节点", unZeroList)
        return unZeroList
    if m < len(unZeroList):
        # 需要进行轮盘选择
        # print("度高节点进行轮盘选择", unZeroList)
        return RouletteChosen(unZeroList)
    if m > len(unZeroList):
        # 随机选择 不是前n个，把zeroList换个顺序
        # print("返回所有度高的节点", unZeroList, "以及部分零度节点", zeroList[0: m - len(unZeroList)])
        return unZeroList + zeroList[0: m - len(unZeroList)]




# 从中心节点开始进行网络建立
def BroadCast(node):

    # print("========================传播开始============")
    # print("当前节点", node.number, "初始度", node.degree)
    linked = []
    if node.degree >= maxDegree:
        return
    # 先将节点周围已经与其连接或达到最大度的节点从邻居列表中删除
    for nei in node.neighbors:
        if (nodeList[nei['no']].degree >= 10) or (node.number in nodeList[nei['no']].linked):
            node.neighbors.remove(nei)
    # 寻找节点进行连接
    linked = BulidLink(node.neighbors)
    
    # 更新本节点度与连接关系
    if len(linked) + node.degree > maxDegree:
        node.linked += linked[0: maxDegree - node.degree]
        node.degree = maxDegree
    else:
        node.linked += linked
        node.degree += len(linked)
        
    node.visited = 1
    
    # print("找到的连接点", linked, "传播后节点度", node.degree)
    
    # 更新相连节点度与连接关系
    for item in linked:
        nodeList[item].degree += 1
        nodeList[item].linked.append(node.number)
                
    # 寻找下一个传播节点
    for nei in node.neighbors:
        if(nodeList[nei['no']].visited == 0):
            # print("由节点", node.number, "传播到", nodeList[nei['no']].number)
            BroadCast(nodeList[nei['no']])
            break
    return 


# 初始节点展示
def ShowInitFig():
    x = []
    y = []
    for node in nodeList:
        x.append(node.location['x'])
        y.append(node.location['y'])
    x = np.array(x)
    y = np.array(y)
    plt.figure(num=1, figsize=(8, 5))   # num为编号，figsize为大小
    plt.scatter(x, y, s=75, alpha=.5)


# 生成网络拓扑
def InitTopo():
    adj = np.zeros((nodeSize, nodeSize), dtype=np.int16)
    for node in nodeList:
        for nei in node.linked:
            adj[node.number, nei] = 1
    graph = nx.from_numpy_matrix(adj)
    return graph


# 建立网络
def BuildNetwork(arg):
    if arg == 'self':       # 用改进的B-A模型生成拓扑结构
        startNode = InitNodes()
        # print("中心节点", startNode.number, "节点位置", startNode.location)
        BroadCast(startNode)

        graph_init = InitTopo()
        R_init = utils.CalRobust(graph_init)
        graph_opti = utils.OptiTopo_hill(graph_init)
        R_opti = utils.CalRobust(graph_opti)

    else:           # 直接生成拓扑结构
        graph_init = nx.barabasi_albert_graph(nodeSize, m)
        R_init = utils.CalRobust(graph_init)
        graph_opti = utils.OptiTopo_hill(graph_init)
        R_opti = utils.CalRobust(graph_opti)

    return graph_init, graph_opti, R_init, R_opti


def InitDataSet(num, mode):
    data_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir)) + '\\data'
    folder = os.path.exists(data_path)
    if not folder:
        os.makedirs(data_path)
    for i in range(num):
        g_init, g_opti, R1, R2 = BuildNetwork(mode)
        nx.write_gpickle(G=g_init, path=os.path.join(data_path, 'init_{}.gpickle'.format(i)))
        nx.write_gpickle(G=g_opti, path=os.path.join(data_path, 'opti_{}.gpickle'.format(i)))


if __name__ == "__main__":
    InitDataSet(1, 'self')
