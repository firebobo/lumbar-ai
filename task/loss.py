import torch


class HeatmapLoss(torch.nn.Module):
    """
    loss for detection heatmap
    """

    def __init__(self):
        super(HeatmapLoss, self).__init__()

    def forward(self, pred, gt):
        heat_l = ((pred - gt) ** 2)
        l = heat_l.sum(dim=3).sum(dim=2).sum(dim=1)
        return l  ## l of dim bsize


class LabelLoss(torch.nn.Module):
    """
    loss for detection heatmap
    """

    def __init__(self):
        super(LabelLoss, self).__init__()

    def forward(self, pred, gt, heatmap):
        # print(gt.shape)
        size = heatmap.shape
        loss = torch.zeros((gt.shape[0]), dtype=float)

        m = size[2]
        n = size[3]
        for idx, g in enumerate(gt):
            l = torch.zeros((gt.shape[1]), dtype=float)
            for jdx, gg in enumerate(g):
                a = heatmap[idx, jdx]
                index = int(a.argmax())
                x = int(index / m)
                y = index % m
                if not a[x, y] == 1:
                    continue
                # xy_loss = (gg[9] + gg[7] - x - pred[idx, jdx,7]) ** 2 + (
                #             gg[10] + gg[8] - y - pred[idx, jdx,8]) ** 2
                # conf_loss = (1 - a[x, y]) ** 2
                class_loss = ((pred[idx, 0:7, x, y] - gg[0:7]) ** 2).sum()
                l[jdx] = class_loss
                # l[jdx] = xy_loss
            loss[idx] = l.sum()
        return loss  ## l of dim bsize


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
        for idx in range(self.nstack + 1):
            combined_loss.append(self.heatmapLoss(combined_preds[idx][:, self.keys:], heatmaps[idx]))
            labels_loss.append(
                self.labelLoss(combined_preds[idx][:, :self.keys], labels, heatmaps[idx]))

        return torch.stack(combined_loss, 1), torch.stack(labels_loss, 1)
