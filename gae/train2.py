from __future__ import division
from __future__ import print_function

import argparse
import time
import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
from model import GCNModelVAE
from optimizer import loss_function
from torch.utils.data import DataLoader
import os
import utils
import networkx as nx
import matplotlib.pyplot as plt


# 模型的训练
def train_gae(args):

    print("============训练开始============")
    # 读取数据集
    init_dir = os.path.join(os.path.pardir, 'data', 'node_100', 'init')
    opti_dir = os.path.join(os.path.pardir, 'data', 'node_100', 'opti')
    train_dataset = utils.PairTopoDataset(init_dir=init_dir, opti_dir=opti_dir)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, num_workers=0)
    # 读取节点数与特征矩阵维度
    # for (cnt, i) in enumerate(train_dataset):
    #     print(i[2])
    #     if cnt == 0:
    #         break
    n_nodes, feat_dim = train_dataset[0][2].shape
    # n_nodes, feat_dim = 100
    # 定义模型与优化器
    model = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    #
    epoch_losses = []
    for epoch in range(args.epochs):
        t = time.time()
        epoch_loss = 0
        for (iter, (adj, adj_norm, features, adj_label)) in enumerate(train_dataloader):
            optimizer.zero_grad()
            adj_pred, mu, logvar = model(features[0], adj_norm[0])
            loss = loss_function(preds=adj_pred, labels=adj_label[0],
                                 mu=mu, logvar=logvar, n_nodes=n_nodes,
                                 adj=adj[0])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
            # print('Epoch: ', epoch, '| Step: ', iter, '| loss: ', loss.detach().item())
        epoch_loss /= (iter + 1)
        # 训练信息输出
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(epoch_loss),
              "time=", "{:.5f}".format(time.time() - t)
              )
        epoch_losses.append(epoch_loss)
    torch.save(model.state_dict(), 'gae2.pkl')
    print("============训练结束============")


# 模型的测试
def test_gae(args):
    print("============测试开始============")
    train_dataset = utils.OptiTopoDataset(data_dir=os.path.join(os.path.pardir, 'data', 'node_100', 'opti'))
    test_dataset = utils.InitTopoDataset(data_dir=os.path.join(os.path.pardir, 'data', 'node_100', 'init'))
    feat_dim = test_dataset[0][2].shape[0]
    model = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
    model.load_state_dict(torch.load('gae2.pkl'))

    model.eval()
    with torch.no_grad():
        adj_pred, mu, logvar = model(test_dataset[6][2], test_dataset[6][1])
        adj_pred = torch.sigmoid(adj_pred)
        print(adj_pred)
        adj_pred = adj_pred.numpy()
        adj_pred[adj_pred < 0.5] = 0
        adj_pred[adj_pred >= 0.5] = 1
        np.fill_diagonal(adj_pred, 0)
        print(test_dataset[2][0])
        print(adj_pred)
        G = nx.from_numpy_matrix(adj_pred)
        G2 = nx.from_numpy_matrix(test_dataset[6][0].numpy())
        print(utils.CalRobust(G))
        print(utils.CalRobust(G2))
        G3 = nx.from_numpy_matrix(train_dataset[6][0].numpy())
        print(utils.CalRobust(G3))

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
        # 显示图表
        plt.show()
    print("============测试结束============")


# 主函数
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
    parser.add_argument('--hidden1', type=int, default=50, help='Number of units in hidden layer 1.')
    parser.add_argument('--hidden2', type=int, default=25, help='Number of units in hidden layer 2.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset-str', type=str, default='BA', help='type of dataset.')
    parser.add_argument("--stage", type=str, default='test', help="is train or test")
    args = parser.parse_args()

    if args.stage == 'train':
        train_gae(args)

    if args.stage == 'test':
        test_gae(args)


if __name__ == '__main__':
    main()

