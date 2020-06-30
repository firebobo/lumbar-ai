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
        l = 0
        for idx,g in enumerate(gt):
            a = heatmap[idx]
            m, n = a.shape
            index = int(a.argmax())
            x = int(index / n)
            y = index % n
            l += ((pred[:,x,y]-g)**2).sum()
        return l ## l of dim bsize