import torch
import torch.nn.modules.loss
import torch.nn.functional as F

#
# def loss_function(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
#     # 输入邻接矩阵与重构邻接矩阵之间的重构损失
#     cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)
#     # q（Z | X，A）与p（Z）之间的KL-散度
#     # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
#     KLD = -0.5 / n_nodes * torch.mean(torch.sum(
#         1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
#     return cost + KLD


def loss_function(preds, labels, mu, logvar, n_nodes, adj):

    pos_weight = torch.Tensor([float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()])
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    # 输入邻接矩阵与重构邻接矩阵之间的重构损失
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)
    # q（Z | X，A）与p（Z）之间的KL-散度
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD
