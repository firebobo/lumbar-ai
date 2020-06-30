import torch

class HeatmapLoss(torch.nn.Module):
    """
    loss for detection heatmap
    """
    def __init__(self):
        super(HeatmapLoss, self).__init__()

    def forward(self, pred, gt):
        l = ((pred - gt)**2)
        l = l.mean(dim=3).mean(dim=2).mean(dim=1)
        return l ## l of dim bsize

class LabelLoss(torch.nn.Module):
    """
    loss for detection heatmap
    """
    def __init__(self):
        super(LabelLoss, self).__init__()

    def forward(self, pred, gt, heatmap):
        # print(gt.shape)
        size = heatmap.shape
        loss = torch.zeros((gt.shape[0]),dtype=float)

        m = size[2]
        n = size[3]
        for idx,g in enumerate(gt):
            l = torch.zeros((gt.shape[1]),dtype=float)
            for jdx,gg in enumerate(g):
                a = heatmap[idx,jdx]
                index = int(a.argmax())
                x = int(index / n)
                y = index % n
                l[jdx] = ((pred[idx,:, x, y] - gg) ** 2).mean()
            loss[idx] = l.mean()
        return loss ## l of dim bsize