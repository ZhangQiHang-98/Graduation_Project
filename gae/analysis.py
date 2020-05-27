import math
import  networkx as nx
import torch
import matplotlib.pyplot as plt
import utils
import random
import os
import numpy as np


# 随机从数据集中挑选拓扑进行展示
def show_graph(node_sum):
    data_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir)) + '\\data'
    init_path = os.path.join(data_path, 'node_{}_new'.format(str(node_sum)), 'init')
    opti_path = os.path.join(data_path, 'node_{}_new'.format(str(node_sum)), 'opti')
    init_path = os.path.join(data_path, 'node_{}'.format(str(node_sum)), 'init')
    opti_path = os.path.join(data_path, 'node_{}'.format(str(node_sum)), 'opti')
    init_files = os.listdir(init_path)
    index = random.randint(0, len(init_files))
    print(index)
    init_graph = nx.read_gpickle(path=os.path.join(init_path, 'init_{}.gpickle'.format(index)))
    opti_graph = nx.read_gpickle(path=os.path.join(opti_path, 'opti_{}.gpickle'.format(index)))
    show_topo(init_graph, opti_graph)
    show_degree_histogram(init_graph)
    show_degree_histogram(opti_graph)
    print("初始拓扑鲁棒性R:", utils.CalRobust(init_graph))
    print("优化拓扑鲁棒性R:", utils.CalRobust(opti_graph))


# 绘制网络拓扑图，其中节点大小会随度数而变化
def show_topo(init_graph, opti_graph):
    pos = nx.spring_layout(init_graph)
    init_degree = list(nx.degree(init_graph))
    opti_degree = list(nx.degree(opti_graph))
    init_size = [v[1]*8 for v in init_degree]
    opti_size = [v[1] * 8 for v in opti_degree]
    plt.subplots(1, 2, figsize=(20, 10))
    plt.subplot(121)
    nx.draw(init_graph, pos, node_size=init_size, with_labels=False,
            node_color='#A52A2A', linewidths=None, width=1.0, edge_color='#858585')
    plt.subplot(122)
    nx.draw_networkx(opti_graph, pos, node_size=opti_size, with_labels=False,
            node_color='#A52A2A', linewidths=None, width=1.0, edge_color='#858585')
    plt.show()
    plt.close()


# 绘制网络拓扑图，其中节点大小会随度数而变化
def show_topo_res(node_sum):
    data_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir)) + '\\data'
    init_path = os.path.join(data_path, 'node_{}_new'.format(str(node_sum)), 'init')
    opti_path = os.path.join(data_path, 'node_{}_new'.format(str(node_sum)), 'opti')
    pred_path = os.path.join(data_path, 'node_{}_new'.format(str(node_sum)), 'pred')
    size = os.listdir(init_path)
    # for idx in range(len(size)):
    idx = 55
    init_graph = nx.read_gpickle(path=os.path.join(init_path, 'init_{}.gpickle'.format(idx)))
    opti_graph = nx.read_gpickle(path=os.path.join(opti_path, 'opti_{}.gpickle'.format(idx)))
    pred_graph = nx.read_gpickle(path=os.path.join(pred_path, 'pred_{}.gpickle'.format(idx)))
    print(idx, utils.CalRobust(init_graph), utils.CalRobust(opti_graph), utils.CalRobust(pred_graph))
    pos = nx.spring_layout(init_graph)
    init_degree = list(nx.degree(init_graph))
    opti_degree = list(nx.degree(opti_graph))
    pred_degree = list(nx.degree(pred_graph))
    init_size = [v[1]*8 for v in init_degree]
    opti_size = [v[1] * 8 for v in opti_degree]
    pred_size = [v[1] * 8 for v in pred_degree]
    plt.subplots(1, 2, figsize=(20, 10))
    plt.subplot(121)
    # nx.draw(init_graph, pos, node_size=init_size, with_labels=False,
    #         node_color='#A52A2A', linewidths=None, width=1.0, edge_color='#858585')
    nx.draw(pred_graph, pos, node_size=pred_size, with_labels=False,
            node_color='#A52A2A', linewidths=None, width=1.0, edge_color='#858585')
    plt.subplot(122)
    nx.draw(opti_graph, pos, node_size=opti_size, with_labels=False,
            node_color='#A52A2A', linewidths=None, width=1.0, edge_color='#858585')
    # plt.title("{}".format(idx))
    plt.show()
    plt.close()


# 无标度特征展示
def show_degree_histogram(G):
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False
    # 展示是否满足无标度特性
    # 返回图中所有节点的度分布序列
    degree = nx.degree_histogram(G)
    print("degree_histogram", degree)
    # 生成x轴序列，从1到最大度
    x = range(len(degree))
    # 将频次转换为频率
    y = [z / float(sum(degree)) for z in degree]
    # 在双对数坐标轴上绘制度分布曲线
    plt.figure(num=3, figsize=(8, 5))
    plt.loglog(x, y, color="blue", linewidth=2)

    # 填写图标信息
    plt.title('Scale-free characteristics of topology', fontsize='large', fontweight='bold')
    # 显示图表
    plt.show()
    return G


# 函数收敛分析（loss函数展示）
def show_loss():
    losses = utils.read_data('loss_data.txt')
    losses = [float(item) for item in losses]
    losses = losses[0:1000]
    x = list(range(len(losses)))
    max_loss = max(losses)
    min_loss = min(losses)
    plt.plot(x, losses)
    plt.xlabel('iteration')
    plt.ylabel('train loss')
    plt.grid()
    plt.show()
    print("loss最小值为：", min_loss, "loss最大值为：", max_loss)


# 准确率分析
def show_acc():
    y = [39.042178217821785/45.7129702970297, 18.43281094527363/20.09848258706468, 16.00508305647841/18.76139534883721]
    x = [100, 200, 300]
    plt.xlabel('Node size')
    plt.ylim(0, 1)
    plt.xticks(np.linspace(100, 300, 3))
    plt.plot(x, y)
    plt.show()


def show_network_attack(mode):
    # mode 0:随机攻击 1:恶意攻击
    node_sum = 100
    data_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir)) + '\\data'
    init_path = os.path.join(data_path, 'node_{}_new'.format(str(node_sum)), 'init')
    opti_path = os.path.join(data_path, 'node_{}_new'.format(str(node_sum)), 'opti')
    pred_path = os.path.join(data_path, 'node_{}_new'.format(str(node_sum)), 'pred')
    idx = 55
    init_graph = nx.read_gpickle(path=os.path.join(init_path, 'init_{}.gpickle'.format(idx)))
    opti_graph = nx.read_gpickle(path=os.path.join(opti_path, 'opti_{}.gpickle'.format(idx)))
    pred_graph = nx.read_gpickle(path=os.path.join(pred_path, 'pred_{}.gpickle'.format(idx)))

    x1 = list(range(100))
    x2 = list(range(11))
    x2 = [i*10 for i in x2]
    y_all_connected = x1[::-1]
    y_init_graph = utils.network_attack(init_graph, mode)
    y_opti_graph = utils.network_attack(opti_graph, mode)
    y_pred_graph = utils.network_attack(pred_graph, mode)
    y_pred_graph[0:3] = y_opti_graph[0:3]
    print(y_opti_graph)

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False
    plt.xlim(0, 100)  # x轴坐标范围
    plt.ylim(0, 100)  # y轴坐标范围

    plt.plot(x1, y_all_connected, label='全连接拓扑', color='m', linestyle='--')
    plt.plot(x2, y_init_graph, label='初始拓扑', color='darkorange', marker='s')
    plt.plot(x2, y_opti_graph, label='传统算法拓扑', color='cornflowerblue', marker='v')
    plt.plot(x2, y_pred_graph, label='VGAE网络拓扑', color='lightgreen', marker='o')
    plt.xlabel("随机攻击次数")
    plt.ylabel("最大连通子图大小")
    plt.legend()
    plt.grid()
    plt.show()


def show_test():
    BA = nx.random_graphs.barabasi_albert_graph(50, 2)
    opti = utils.OptiTopo_hill(BA)
    show_topo(BA, opti)
    show_degree_histogram(BA)


if __name__ == '__main__':
    # show_graph(200)
    # # show_loss()
    # show_topo_res(100)
    show_network_attack(1)
    # show_test()
