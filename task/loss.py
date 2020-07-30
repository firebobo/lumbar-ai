import torch


class HeatmapLoss(torch.nn.Module):
    """
    loss for detection heatmap
    """

    def __init__(self):
        super(HeatmapLoss, self).__init__()

    def forward(self, pred, gt):
        heat_l = ((pred - gt) ** 2)
        l = heat_l.sum(dim=3).sum(dim=2).mean(dim=1)
        return l  ## l of dim bsize
        # # print(gt.shape)
        # size = pred.shape
        # loss = torch.zeros((size[0]), dtype=float)
        #
        # m = size[2]
        # n = size[3]
        # for idx, g in enumerate(gt):
        #     l = torch.zeros((size[1]), dtype=float)
        #     for jdx,gg in enumerate(g):
        #         l[jdx] = torch.nn.BCELoss(reduction='sum')(pred[idx,jdx], gg)
        #     loss[idx] = l.mean()
        # return loss  ## l of dim bsize

class LabelLoss(torch.nn.Module):
    """
    loss for detection heatmap
    """

    def __init__(self,key):
        super(LabelLoss, self).__init__()
        self.key = key

    def forward(self, pred, gt, heatmap):
        # print(gt.shape)
        size = heatmap.shape
        loss = torch.zeros((gt.shape[0]), dtype=float)

        m = size[2]
        n = size[3]
        heatmap = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(heatmap)
        for idx, g in enumerate(gt):
            l = torch.zeros((gt.shape[1]), dtype=float)
            for jdx, gg in enumerate(g):
                a = heatmap[idx, jdx]
                index = int(a.argmax())
                x = int(index / m)
                y = index % m
                # if not a[x, y] == 1:
                #     continue
                # xy_loss = (gg[9] + gg[7] - x - pred[idx, jdx,7]) ** 2 + (
                #             gg[10] + gg[8] - y - pred[idx, jdx,8]) ** 2
                # conf_loss = (1 - a[x, y]) ** 2
                # class_loss = ((pred[idx, 0:7, x, y] - gg[0:7]) ** 2).sum()
                class_loss = torch.nn.MSELoss(reduction='sum')(pred[idx, jdx*7:jdx*7+7, x, y], gg[0:7])
                l[jdx] = class_loss
                # l[jdx] = xy_loss
            loss[idx] = l.mean()
        return loss  ## l of dim bsize


class MaskLoss(torch.nn.Module):
    """
    loss for detection heatmap
    """

    def __init__(self):
        super(MaskLoss, self).__init__()

    def forward(self, pred, gt):
        # print(gt.shape)
        size = pred.shape
        loss = torch.zeros((size[0]), dtype=float)

        m = size[2]
        n = size[3]
        for idx, g in enumerate(gt):
            l = torch.zeros((size[1]), dtype=float)
            for jdx,gg in enumerate(g):
                l[jdx] = torch.nn.MSELoss(reduction='sum')(pred[idx,jdx], gg)
            loss[idx] = l.mean()
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
        self.labelLoss = LabelLoss(inference_keys)
        self.maskLoss = MaskLoss()

    def forward(self, combined_preds, heatmaps, labels, masks):
        combined_loss = []
        labels_loss = []
        masks_loss = []

        for idx in range(self.nstack + 1):
            heatmap_preds = combined_preds[idx][:, :self.keys]*masks[idx]
            combined_loss.append(self.heatmapLoss(heatmap_preds, heatmaps[idx]))
            labels_loss.append(
                self.labelLoss(combined_preds[idx][:, 2*self.keys:], labels, heatmap_preds))
            masks_loss.append(
                self.maskLoss(combined_preds[idx][:, self.keys:2*self.keys],masks[idx]))

        return torch.stack(combined_loss, 1), torch.stack(labels_loss, 1), torch.stack(masks_loss, 1)
