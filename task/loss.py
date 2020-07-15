import torch

class HeatmapLoss(torch.nn.Module):
    """
    loss for detection heatmap
    """
    def __init__(self):
        super(HeatmapLoss, self).__init__()

    def forward(self, pred, gt):
        heat_l = ((pred - gt)**2)
        l = heat_l.mean(dim=3).mean(dim=2).sum(dim=1)
        return l ## l of dim bsize

class LabelLoss(torch.nn.Module):
    """
    loss for detection heatmap
    """
    def __init__(self):
        super(LabelLoss, self).__init__()

    def forward(self, pred, gt):
        # print(gt.shape)
        loss = torch.zeros((gt.shape[0]),dtype=float)
        for idx,g in enumerate(gt):
            l = torch.zeros((gt.shape[1]),dtype=float)
            for jdx,gg in enumerate(g):
                # if gg[9] <= 0 or gg[10] <= 0 or gg[9] >= m or gg[10] >= n:
                #     continue
                # a = heatmap[idx,jdx]
                # index = int(a.argmax())
                # x = int(index / m)
                # y = index % m
                # xy_loss = ((gg[9]+gg[7] - x-pred[idx,7, jdx, 0])/n)**2+((gg[10] +gg[8]- y-pred[idx, 8, jdx, 0])/m)**2
                # conf_loss = (1 - a[x, y])**2
                class_loss = ((pred[idx, jdx, 0:7] - gg[0:7]) ** 2).sum()
                # l[jdx] = class_loss + xy_loss + conf_loss
                l[jdx] = class_loss
            loss[idx] = l.sum()
        return loss ## l of dim bsize


class KeypointLoss(torch.nn.Module):
    """
    The wrapper module that will behave differetly for training or testing
    inference_keys specify the inputs for inference
    """

    def __init__(self, inference_keys, nstack, nclass):
        super(KeypointLoss, self).__init__()
        self.keys = inference_keys
        self.nstack = nstack
        self.n_class = nclass
        self.heatmapLoss = HeatmapLoss()
        self.labelLoss = LabelLoss()

    def forward(self, combined_preds, heatmaps, labels):
        combined_loss = []
        labels_loss = []
        for idx in range(self.nstack+1):
            combined_loss.append(self.heatmapLoss(combined_preds[0][idx], heatmaps[idx].cuda()))
            labels_loss.append(self.labelLoss(combined_preds[1][idx], labels))

        return torch.stack(combined_loss,1), torch.stack(labels_loss,1)
