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
import analysis


# 模型的训练
def train_gae(args):

    print("============训练开始============")
    # 读取数据集
    train_dataset = utils.OptiTopoDataset(data_dir=os.path.join(os.path.pardir, 'data', 'node_300', 'opti'))
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, num_workers=0)
    # 读取节点数与特征矩阵维度
    # for (cnt, i) in enumerate(train_dataset):
    #     print(i[2])
    #     if cnt == 0:
    #         break
    n_nodes, feat_dim = train_dataset[0][2].shape
    # n_nodes, feat_dim = 100
    # # 得到原始的邻接矩阵与特征矩阵(X,此处暂为单位矩阵)
    # adj, features = utils.load_data(args.dataset_str)
    # # print(type(adj))
    # n_nodes, feat_dim = features.shape
    # # # 保留原始邻接矩阵（对角线没填0）,移除对角元素，保持对角线上为0
    # # adj_orig = adj
    # # adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    # # adj_orig.eliminate_zeros()
    # # assert np.diag(adj_orig.todense()).sum() == 0
    #
    # # 正则化之后的A~
    # adj_norm = utils.preprocess(adj)
    #
    # # 标签A(对角线为1)
    # adj_label = adj + sp.eye(adj.shape[0])
    # adj_label = torch.FloatTensor(adj_label.toarray())
    #
    # 定义损失函数中的pos_weight与norm参数
    # pos_weight = torch.Tensor([float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()])
    # norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
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
            epoch_losses.append(loss.detach().item())
            # print('Epoch: ', epoch, '| Step: ', iter, '| loss: ', loss.detach().item())
        epoch_loss /= (iter + 1)
        # 训练信息输出
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(epoch_loss),
              "time=", "{:.5f}".format(time.time() - t)
              )

    utils.data_write(epoch_losses)

    torch.save(model.state_dict(), 'gae_300.pkl')
    print("============训练结束============")


# 模型的测试
def test_gae(args):
    test_dataset = utils.InitTopoDataset(data_dir=os.path.join(os.path.pardir, 'data', 'node_100_new', 'init'))
    pred_data_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir, 'data', 'node_100_new', 'pred'))
    # 加载模型
    feat_dim = test_dataset[0][2].shape[0]
    model = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
    model.load_state_dict(torch.load('gae.pkl'))
    model.eval()
    count = 0
    with torch.no_grad():
        print("============测试开始============")
        for topo in test_dataset:
            print(1)
            if count >= len(test_dataset)-1:
                break
            adj_pred, mu, logvar = model(topo[2], topo[1])
            adj_pred = utils.transfer_mat(adj_pred)
            G_pred = nx.from_numpy_matrix(adj_pred)
            nx.write_gpickle(G=G_pred, path=os.path.join(pred_data_path, 'pred_{}.gpickle'.format(count)))
            count += 1
    print("============测试结束============")



# 主函数
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
    parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
    parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset-str', type=str, default='BA', help='type of dataset.')
    parser.add_argument("--stage", type=str, default='test', help="is train or test")
    args = parser.parse_args()

    if args.stage == 'train':
        train_gae(args)

    if args.stage == 'test':
        test_gae(args)


if __name__ == '__main__':
    main()

