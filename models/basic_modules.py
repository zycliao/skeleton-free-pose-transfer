import torch
import numpy as np
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from torch.nn import Sequential, Dropout, Linear, ReLU, BatchNorm1d, Parameter


def MLP(channels, batch_norm=True):
    if batch_norm:
        return Sequential(*[Sequential(Linear(channels[i - 1], channels[i]), ReLU(), BatchNorm1d(channels[i], momentum=0.1))
                            for i in range(1, len(channels))])
    else:
        return Sequential(*[Sequential(Linear(channels[i - 1], channels[i]), ReLU()) for i in range(1, len(channels))])


class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels, nn, aggr='max', **kwargs):
        super(EdgeConv, self).__init__(aggr=aggr, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nn = nn

    def forward(self, x, edge_index):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        return self.nn(torch.cat([x_i, (x_j - x_i)], dim=1))

    def update(self, aggr_out):
        aggr_out = aggr_out.view(-1, self.out_channels)
        return aggr_out

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)



class GCUTPL(torch.nn.Module):
    def __init__(self, in_channels, out_channels, aggr='max'):
        super(GCUTPL, self).__init__()
        self.edge_conv_tpl = EdgeConv(in_channels=in_channels, out_channels=out_channels,
                                      nn=MLP([in_channels * 2, out_channels, out_channels]), aggr=aggr)
        # self.edge_conv_geo = EdgeConv(in_channels=in_channels, out_channels=out_channels // 2,
        #                           nn=MLP([in_channels * 2, out_channels // 2, out_channels // 2]), aggr=aggr)
        self.mlp = MLP([out_channels, out_channels])

    def forward(self, x, tpl_edge_index):
        x_tpl = self.edge_conv_tpl(x, tpl_edge_index)
        x_out = self.mlp(x_tpl)
        return x_out
