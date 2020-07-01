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
            labels_loss.append(self.labelLoss(combined_lb_preds[:, i], labels, heatmaps))

        combined_loss = torch.stack(combined_loss, dim=1)
        labels_loss = torch.stack(labels_loss, dim=1)
        return combined_loss, labels_loss