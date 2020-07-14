import torch
from torch import nn
from models.layers import Conv, Hourglass, Pool, Residual, ClsHead
from task.loss import HeatmapLoss, LabelLoss


class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(-1, 256, 4, 4)


class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class PoseNet(nn.Module):
    def __init__(self, nstack, inp_dim, oup_dim, num_class=9, bn=False, increase=0, **kwargs):
        super(PoseNet, self).__init__()

        self.nstack = nstack
        self.pre = nn.Sequential(
            Conv(1, 64, 7, 2, bn=True, relu=True),
            Residual(64, 128),
            Pool(2, 2),
            Residual(128, 128),
            Residual(128, inp_dim)
        )

        self.hgs = nn.ModuleList([
            nn.Sequential(
                Hourglass(4, inp_dim, bn, increase),
            ) for i in range(nstack)])

        self.features = nn.ModuleList([
            nn.Sequential(
                Residual(inp_dim, inp_dim),
                Residual(inp_dim, inp_dim),
                Conv(inp_dim, inp_dim, 1, bn=True, relu=False)
            ) for i in range(nstack)])

        self.outs_heatmap = nn.ModuleList([Conv(inp_dim, oup_dim+oup_dim, 1, relu=True, bn=True) for i in range(nstack)])

        self.merge_features = nn.ModuleList([Merge(inp_dim, inp_dim) for i in range(nstack - 1)])
        self.merge_preds = nn.ModuleList([Merge(oup_dim+oup_dim, inp_dim) for i in range(nstack - 1)])
        self.nstack = nstack

    def forward(self, imgs):
        ## our posenet
        # x = imgs.permute(0, 3, 1, 2) #x of size 1,3,inpdim,inpdim
        x = self.pre(imgs)
        combined_hm_preds = []
        for i in range(self.nstack):
            hg = self.hgs[i](x)
            feature = self.features[i](hg)
            preds = self.outs_heatmap[i](feature)
            combined_hm_preds.append(preds)
            if i < self.nstack - 1:
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)

        return torch.stack(combined_hm_preds, 1)
