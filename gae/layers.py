import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


class GraphConvolution(Module):
    """
    GCN卷积层实现，包括dropout与激活函数（与论文相同，使用Relu）
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    # 利用kaiming_uniform_方法对权重初始化(在Relu函数中表现得较好)
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight)

    def forward(self, input, adj):
        # 防止过拟合，调用dropout方法
        input = F.dropout(input, self.dropout, self.training)
        # 指代H(l)与W相乘
        support = torch.mm(input, self.weight)
        # 正则化之后的邻接矩阵和前面的结果相乘,spmm针对稀疏矩阵
        output = torch.spmm(adj, support)
        # 激活函数
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'