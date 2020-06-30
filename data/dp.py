import cv2
import sys
import os
import torch
import numpy as np
import torch.utils.data
import utils.img
import data.ref as ds

class GenerateHeatmap():
    def __init__(self, output_res, num_parts):
        self.output_res = output_res
        self.num_parts = num_parts
        sigma = self.output_res/64
        self.sigma = sigma
        size = 6*sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, keypoints):
        hms = np.zeros(shape = (self.num_parts, self.output_res, self.output_res), dtype = np.float32)
        sigma = self.sigma
        for p in keypoints:
            for idx, pt in enumerate(p):
                print(p)
                if pt[0] > 0: 
                    x, y = int(pt[0]), int(pt[1])
                    if x<0 or y<0 or x>=self.output_res or y>=self.output_res:
                        continue
                    ul = int(x - 3*sigma - 1), int(y - 3*sigma - 1)
                    br = int(x + 3*sigma + 2), int(y + 3*sigma + 2)

                    c,d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                    a,b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                    cc,dd = max(0, ul[0]), min(br[0], self.output_res)
                    aa,bb = max(0, ul[1]), min(br[1], self.output_res)
                    hms[idx, aa:bb,cc:dd] = np.maximum(hms[idx, aa:bb,cc:dd], self.g[a:b,c:d])
        return hms

class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, ds, index):
        self.input_res = config['train']['input_res']
        self.output_res = config['train']['output_res']
        self.generateHeatmap = GenerateHeatmap(self.output_res, config['inference']['num_parts'])
        self.ds = ds
        self.index = index

    def __len__(self):
        return self.index

    def __getitem__(self, idx):
        return self.loadImage(idx)

    def loadImage(self, idx):
        ds = self.ds

        orig_img = ds.get_img(idx)
        keypoints = ds.get_kps(idx)
        labels = ds.get_labels(idx)

        
        ## generate heatmaps on outres
        heatmaps = self.generateHeatmap(keypoints)
        
        return orig_img.astype(np.float32), heatmaps.astype(np.float32),np.array(labels)

    def preprocess(self, data):
        # random hue and saturation
        data = cv2.cvtColor(data, cv2.COLOR_RGB2HSV);
        delta = (np.random.random() * 2 - 1) * 0.2
        data[:, :, 0] = np.mod(data[:,:,0] + (delta * 360 + 360.), 360.)

        delta_sature = np.random.random() + 0.5
        data[:, :, 1] *= delta_sature
        data[:,:, 1] = np.maximum( np.minimum(data[:,:,1], 1), 0 )
        data = cv2.cvtColor(data, cv2.COLOR_HSV2RGB)

        # adjust brightness
        delta = (np.random.random() * 2 - 1) * 0.3
        data += delta

        # adjust contrast
        mean = data.mean(axis=2, keepdims=True)
        data = (data - mean) * (np.random.random() + 0.5) + mean
        data = np.minimum(np.maximum(data, 0), 1)
        return data


def init(config):
    batchsize = config['train']['batchsize']
    current_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_path)
    train = config['inference']['train_num_eval']


    dataset = { key: Dataset(config, ds.Lumbar(), data) for key, data in zip( ['train'], [int(train)] ) }

    use_data_loader = config['train']['use_data_loader']

    loaders = {}
    for key in dataset:
        loaders[key] = torch.utils.data.DataLoader(dataset[key], batch_size=batchsize, shuffle=True, num_workers=config['train']['num_workers'], pin_memory=False)

    def gen(phase):
        batchsize = config['train']['batchsize']
        batchnum = config['train']['{}_iters'.format(phase)]
        loader = loaders[phase].__iter__()
        for i in range(batchnum):
            imgs, heatmaps = next(loader)
            yield {
                'imgs': imgs, #cropped and augmented
                'heatmaps': heatmaps, #based on keypoints. 0 if not in img for joint
            }


    return lambda key: gen(key)
