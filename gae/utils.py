import networkx as nx
import os
import matplotlib.pyplot as plt
import random
import numpy as np
import math
import operator
import torch
import scipy.sparse as sp
from torch.utils.data import Dataset
import xlwt


# 计算最大连通子图函数，返回的是连通子图（按大小依次排列）
def connected_component_subgraphs(G):
    for c in nx.connected_components(G):
        yield G.subgraph(c)


# 对拓扑进行随即攻击
def network_attack(G, mode):
    test_G = nx.Graph(G)
    N = nx.number_of_nodes(test_G)
    node_list = list(test_G.nodes())
    point = [x*10 for x in list(range(11))]
    print(point)
    res = []
    # 进行N次攻击，分为随机攻击和恶意攻击两种
    if mode == 0:
        while len(node_list) > 0:
            # graphs代表了所有连通子图
            graphs = list(connected_component_subgraphs(test_G))
            if len(node_list) in point:
                res.append(len(graphs[0]))
            selected_node = random.choice(node_list)
            node_list.remove(selected_node)
            test_G.remove_node(selected_node)
        res.append(0)
    else:
        for i in range(N):
            # graphs代表了所有连通子图
            graphs = list(connected_component_subgraphs(test_G))
            # 删除当前度最高的节点
            degree_list = list(test_G.degree())
            del_node = sorted(degree_list, key=lambda x: (x[1], x[0]), reverse=True)[0][0]
            if i in point:
                res.append(len(graphs[0]))
            test_G.remove_node(del_node)
        res.append(0)
    return res


# 计算鲁棒衡量系数R
def CalRobust(G):
    test_G = nx.Graph(G)
    numerator = 0
    N = nx.number_of_nodes(test_G)
    # test_G用于计算R
    # 进行N次恶意攻击（N为节点总数）
    for i in range(N):
        # graphs代表了所有连通子图
        graphs = list(connected_component_subgraphs(test_G))
        numerator += len(graphs[0])
        # 删除当前度最高的节点
        degreeList = list(test_G.degree())
        delNode = sorted(degreeList, key=lambda x: (x[1], x[0]), reverse=True)[0][0]
        test_G.remove_node(delNode)
    # 进行计算
    return numerator / (N * (N + 1))


# 判断重组边是否已经存在
def has_path(G, a, b):
    edges = list(G.edges())
    if (a, b) in edges or (b, a) in edges:
        return 0
    else:
        return 1


# 返回随即交换完的拓扑结构
def switch_edges(G):
    new_G = nx.Graph(G)
    # print(new_G.edges())
    edges = list(new_G.edges())
    # 随机选择两条边
    rand = np.random.randint(0, len(edges), 2)
    # 选择交换方式
    rand_mode = random.choice([0, 1])
    edge1 = edges[rand[0]]
    edge2 = edges[rand[1]]
    # 待交换的四个点
    a, b, c, d = edge1[0], edge1[1], edge2[0], edge2[1]
    # print('Before------edge:', edge1, edge2, 'degree:', new_G.degree([a, b, c, d]))
    new_edge1, new_edge2 = [], []
    # 如果两边有四点的flag
    repeat_flag = a not in edge2 and b not in edge2
    # 判断重组边是否已经存在的标志
    path_flag = has_path(new_G, a, c) and has_path(new_G, a, d) and has_path(new_G, b, c) and has_path(new_G, b, d)
    if repeat_flag and path_flag:

        new_G.remove_edge(a, b)
        new_G.remove_edge(c, d)
        if rand_mode:
            # new_edge1+=[edge1[0], edge2[0]]
            # new_edge2+=[edge1[1], edge2[1]]
            new_G.add_edge(a, c)
            new_G.add_edge(b, d)
        else:
            # new_edge1+=[edge1[0], edge2[1]]
            # new_edge2+=[edge1[1], edge2[0]]
            new_G.add_edge(a, d)
            new_G.add_edge(b, c)

    # print('After------edge:', new_edge1, new_edge2, 'newG.degree:', new_G.degree([a, b, c, d]))

    return new_G


# 鲁棒性优化函数（模拟退火法）
def OptiTopo_sa(graph_init):
    G = nx.Graph(graph_init)
    print("初始R值", CalRobust(G))
    # 设置初始化温度与最终温度,每个温度下的迭代次数
    T = 1
    Tmin = 1e-3
    k = 10
    while T > Tmin:
        print(T)
        for i in range(k):
            new_G = switch_edges(G)
            R_old = CalRobust(G)
            R_new = CalRobust(new_G)
            if R_new > R_old:
                G = nx.Graph(new_G)
            else:
                p = math.exp(-abs(R_old - R_new) / T)
                r = np.random.uniform(low=0, high=1)
                if r < p:
                    G = nx.Graph(new_G)
        # 降温
        T *= 0.9

    print("结束后的R值", CalRobust(G))
    return G


# 鲁棒性优化函数（爬山算法）
def OptiTopo_hill(graph_init):

    G = nx.Graph(graph_init)
    k = 100
    print("初始R值", CalRobust(G))
    # 设置初始化温度与最终温度,每个温度下的迭代次数
    for i in range(k):
        new_G = switch_edges(G)
        R_old = CalRobust(G)
        R_new = CalRobust(new_G)
        if R_new > R_old:
            G = nx.Graph(new_G)
    print("结束后的R值", CalRobust(G))
    return G


# 加载数据集，返回原始邻接矩阵和特征矩阵  -----------需要填写完整
def load_data(G):
    adj = nx.adjacency_matrix(G)
    features = torch.eye(adj.shape[0])
    return adj, features


# 将一个稀疏矩阵转为稀疏张量矩阵
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# 对邻接矩阵进行预处理，由A变为A~
def preprocess(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_normalized


# 初始拓扑数据集
class InitTopoDataset(Dataset):
    def __init__(self, data_dir):
        self.init_data_dir = data_dir
        self.size = len(os.listdir(data_dir))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        data_path = self.init_data_dir
        init_graph = nx.read_gpickle(path=os.path.join(data_path, 'init_{}.gpickle'.format(idx)))
        # 取拓扑的邻接矩阵与特征矩阵
        adj, features = load_data(init_graph)
        # 正则化之后的A~
        adj_norm = preprocess(adj)
        adj_norm = torch.FloatTensor(adj_norm.toarray())
        # 标签A(对角线为1)
        adj_label = adj + sp.eye(adj.shape[0])
        adj_label = torch.FloatTensor(adj_label.toarray())
        adj = torch.FloatTensor(adj.toarray())
        sample = {'adj': adj, 'adj_norm': adj_norm, 'features': features, 'adj_label': adj_label}
        # 返回的样本
        return adj, adj_norm, features, adj_label


# 优化拓扑数据集
class OptiTopoDataset(Dataset):
    def __init__(self, data_dir):
        self.init_data_dir = data_dir
        self.size = len(os.listdir(data_dir))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        data_path = self.init_data_dir
        opti_graph = nx.read_gpickle(path=os.path.join(data_path, 'opti_{}.gpickle'.format(idx)))
        # 取拓扑的邻接矩阵与特征矩阵
        adj, features = load_data(opti_graph)
        # 正则化之后的A~
        adj_norm = preprocess(adj)
        adj_norm = torch.FloatTensor(adj_norm.toarray())
        # 标签A(对角线为1)
        adj_label = adj + sp.eye(adj.shape[0])
        adj_label = torch.FloatTensor(adj_label.toarray())
        adj = torch.FloatTensor(adj.toarray())
        sample = {'adj': adj, 'adj_norm': adj_norm, 'features': features, 'adj_label': adj_label}
        # 返回的样本
        return adj, adj_norm, features, adj_label


# 成对拓扑数据集
class PairTopoDataset(Dataset):
    def __init__(self, init_dir, opti_dir):
        self.init_data_dir = init_dir
        self.opti_data_dir = opti_dir
        self.size = len(os.listdir(init_dir))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        init_data_path = self.init_data_dir
        opti_data_path = self.opti_data_dir
        init_graph = nx.read_gpickle(path=os.path.join(init_data_path, 'init_{}.gpickle'.format(idx)))
        opti_graph = nx.read_gpickle(path=os.path.join(opti_data_path, 'opti_{}.gpickle'.format(idx)))
        # 取拓扑的邻接矩阵与特征矩阵
        adj_init, features = load_data(init_graph)
        adj_opti, features = load_data(opti_graph)
        # 正则化之后的A~
        adj_norm = preprocess(adj_init)
        adj_norm = torch.FloatTensor(adj_norm.toarray())
        # 标签A(对角线为1)
        adj_label = adj_opti + sp.eye(adj_opti.shape[0])
        adj_label = torch.FloatTensor(adj_label.toarray())
        adj = torch.FloatTensor(adj_opti.toarray())
        sample = {'adj': adj, 'adj_norm': adj_norm, 'features': features, 'adj_label': adj_label}
        # 返回的样本
        return adj, adj_norm, features, adj_label
        return init_graph


# 将数据写入新文件
def data_write(datas):
    f = open('loss_data.txt', 'w')
    for loss in datas:
        f.write(str(loss))
        f.write('\n')
    f.close()


# 文件读取
def read_data(file_name):
    loss = []
    with open(file_name, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')  # 去掉列表中每一个元素的换行符
            loss.append(line)
    return loss


# 矩阵转换
def transfer_mat(adj):
    adj = torch.sigmoid(adj)
    adj = adj.numpy()
    if adj.shape[0] == 100:
        threshold = 0.78
    if adj.shape[0] == 200:
        threshold = 0.80
    if adj.shape[0] == 300:
        threshold = 0.82
    adj[adj < threshold] = 0
    adj[adj >= threshold] = 1
    np.fill_diagonal(adj, 0)
    return adj
