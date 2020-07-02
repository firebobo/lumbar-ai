import torch

class HeatmapLoss(torch.nn.Module):
    """
    loss for detection heatmap
    """
    def __init__(self):
        super(HeatmapLoss, self).__init__()

    def forward(self, pred, gt):
        l = ((pred - gt)**2)
        l = l.sum(dim=3).sum(dim=2).sum(dim=1)
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
                if gg[9] < 0 or gg[10] < 0 or gg[9] >= m or gg[10] >= n:
                    continue
                a = heatmap[idx,jdx]
                index = int(a.argmax())
                x = int(index / m)
                y = index % m
                xy_loss = (gg[9] - x)**2+(gg[10] - y)**2
                conf_loss = (1 - a[x, y])**2
                class_loss = ((pred[idx, :, x, y] - gg[0:9]) ** 2).sum()
                l[jdx] = class_loss + xy_loss + conf_loss
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

    def forward(self, combined_hm_preds, combined_lb_preds, **gt):
        loss = self.calc_loss(combined_hm_preds=combined_hm_preds, combined_lb_preds=combined_lb_preds, **gt)
        return loss

    def calc_loss(self, combined_hm_preds, combined_lb_preds, heatmaps, labels):
        combined_loss = []
        labels_loss = []
        for i in range(self.nstack):
            combined_loss.append(self.heatmapLoss(combined_hm_preds[:, i], heatmaps))
            labels_loss.append(self.labelLoss(combined_lb_preds[:, i], labels, combined_hm_preds[:, i]))

        combined_loss = torch.stack(combined_loss, dim=1)
        labels_loss = torch.stack(labels_loss, dim=1)
        return combined_loss, labels_loss