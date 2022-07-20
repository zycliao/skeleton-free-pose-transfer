import torch
from torch_scatter import scatter_max, scatter_sum, scatter_softmax
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d
from models.basic_modules import MLP, GCUTPL


class HandlePredictorSWTpl(torch.nn.Module):
    def __init__(self, input_dim, num_part, aggr='max'):
        super(HandlePredictorSWTpl, self).__init__()
        self.gcu_1 = GCUTPL(in_channels=input_dim, out_channels=64, aggr=aggr)
        self.gcu_2 = GCUTPL(in_channels=64, out_channels=128, aggr=aggr)
        self.gcu_3 = GCUTPL(in_channels=128, out_channels=256, aggr=aggr)
        self.mlp_glb = MLP([(64 + 128 + 256), 256])
        self.mlp2 = MLP([256, 256, 128])
        self.mlp3 = Linear(128, num_part)

    def forward(self, x, tpl_edge_index=None, batch=None, data=None, verbose=False):
        """
        treat the output as skinning weight instead of heatmap
        score: (N_all, K)
        weighted_pos: (bs, K, 3)
        """
        if data is not None:
            tpl_edge_index = data.tpl_edge_index
            batch = data.batch
        pos = x[:, :3]
        x_1 = self.gcu_1(x, tpl_edge_index)
        x_2 = self.gcu_2(x_1, tpl_edge_index)
        x_3 = self.gcu_3(x_2, tpl_edge_index)
        x_4 = self.mlp_glb(torch.cat([x_1, x_2, x_3], dim=1))
        x = self.mlp2(x_4)
        x = self.mlp3(x)
        # softmax
        skinning_weights = torch.softmax(x, 1)
        score = skinning_weights / torch.repeat_interleave(scatter_sum(skinning_weights, batch, dim=0),
                                                           torch.bincount(batch), dim=0)

        weighted_pos = score[:, :, None] * pos[:, None]
        weighted_pos = scatter_sum(weighted_pos, batch, dim=0)

        return score, weighted_pos, x, skinning_weights


class PerPartEncoderTpl(torch.nn.Module):
    # compared to ShapeEncoder, this proviced more flexible interfaces
    # it is used in train_1214.py
    def __init__(self, input_dim, output_dim, aggr='max'):
        super(PerPartEncoderTpl, self).__init__()
        self.gcu_1 = GCUTPL(in_channels=input_dim, out_channels=64, aggr=aggr)
        self.gcu_2 = GCUTPL(in_channels=64, out_channels=128, aggr=aggr)
        self.gcu_3 = GCUTPL(in_channels=128, out_channels=256, aggr=aggr)
        self.mlp_1 = MLP([(64 + 128 + 256), 256])
        self.mlp_glb = MLP([(64 + 128 + 256), 256])

        self.linear = Linear(512, output_dim)
        self.relu = ReLU()
        self.norm = BatchNorm1d(output_dim, momentum=0.1)

    def forward(self, pos, hm, tpl_edge_index=None, batch=None, data=None, feat=None):
        """
        hm: part heat map. (B*V, K). Be careful of the normalization (not skinning weights)
        output: (B, K, C)
        """
        if data is not None:
            tpl_edge_index = data.tpl_edge_index
            batch = data.batch
        if feat is not None:
            x_in = torch.cat((pos, feat), 1)
        else:
            x_in = pos
        x_1 = self.gcu_1(x_in, tpl_edge_index)
        x_2 = self.gcu_2(x_1, tpl_edge_index)
        x_3 = self.gcu_3(x_2, tpl_edge_index)
        x_123 = torch.cat([x_1, x_2, x_3], dim=1)
        x_4 = self.mlp_1(x_123)  # (B*V, 256)
        x_global, _ = scatter_max(x_123, batch, dim=0)  # (B, C)
        x_global = self.mlp_glb(x_global)
        x_global = torch.repeat_interleave(x_global, torch.bincount(batch), dim=0)  # (B*V, 256)

        x_5 = torch.cat((x_4, x_global), 1) # (B*V, 512)
        x_6 = scatter_sum(x_5[:, None] * hm[:, :, None], batch, dim=0)  # (B, K, 512)
        x_6 = self.linear(x_6)
        x_6 = self.relu(x_6)
        x_6 = x_6.permute(0, 2, 1)
        y = self.norm(x_6).permute(0, 2, 1)
        return y


class PerPartDecoder(torch.nn.Module):
    """
    predict (pos, quat)
    """
    def __init__(self, input_dim):
        super(PerPartDecoder, self).__init__()
        net = [Linear(input_dim, 256), ReLU(inplace=True),
               Linear(256, 128), ReLU(inplace=True),
               Linear(128, 128), ReLU(inplace=True),
               Linear(128, 7)]
        self.net = Sequential(*net)

    def forward(self, *feat, base=None):
        # feat: (B, K, C)
        x = torch.cat(feat, 2)
        x = self.net(x)
        if base is not None:
            x = x + base
        return x
