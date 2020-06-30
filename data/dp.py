import cv2
import sys
import os
import torch
import numpy as np
from PIL import Image
import torch.utils.data
from torchvision.transforms import Compose, Resize, ToTensor

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
        for idx, pt in enumerate(keypoints):
            if pt[0] > 0:
                x, y = int(pt[0]), int(pt[1])
                if x < 0 or y < 0 or x >= self.output_res or y >= self.output_res:
                    continue
                ul = int(x - 3 * sigma - 1), int(y - 3 * sigma - 1)
                br = int(x + 3 * sigma + 2), int(y + 3 * sigma + 2)

                c, d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                a, b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                cc, dd = max(0, ul[0]), min(br[0], self.output_res)
                aa, bb = max(0, ul[1]), min(br[1], self.output_res)
                hms[idx, aa:bb, cc:dd] = np.maximum(hms[idx, aa:bb, cc:dd], self.g[a:b, c:d])
        return hms

class GenerateLabelmap():
    def __init__(self, output_res, num_class):
        self.output_res = output_res
        self.num_class = num_class


    def __call__(self, keypoints,labels):
        hms = np.zeros(shape = (self.num_class, self.output_res, self.output_res), dtype = np.float32)
        return hms

class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, ds,size, index,transforms=None):
        self.input_res = config['train']['input_res']
        self.output_res = config['train']['output_res']
        self.size = size
        self.generateHeatmap = GenerateHeatmap(self.output_res, config['inference']['num_parts'])
        self.ds = ds
        self.index = index
        if transforms == None:
          self.transforms = Compose([Resize((self.input_res, self.input_res)), ToTensor()])
        else:
          self.transforms = transforms

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.loadImage(idx%self.index)

    def loadImage(self, idx):
        ds = self.ds

        orig_img = ds.get_img(idx)
        keypoints = ds.get_kps(idx)
        labels = ds.get_labels(idx)

        
        ## generate heatmaps on outres
        heatmaps = self.generateHeatmap(keypoints)
        img = self.transforms(Image.fromarray(orig_img))
        return img, heatmaps.astype(np.float32),np.array(labels).astype(np.float32)

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

    annot_path = r'/annotation.json'
    train_data_dir = r'/home/dwxt/project/dcm/train'
    valid_data_dir = r'/home/dwxt/project/dcm/valid'
    info_name = r'/info.csv'
    size = config['train']['epoch_num'] * config['train']['data_num']

    train_db = Dataset(config, ds.Lumbar(train_data_dir, annot_path, info_name), size, 150)
    valid_db = Dataset(config, ds.Lumbar(valid_data_dir, annot_path, info_name), size, 51)

    dataset = {'train':train_db,'valid':valid_db}

    loaders = {}
    for key in dataset:
        loaders[key] = torch.utils.data.DataLoader(dataset[key], batch_size=batchsize, shuffle=True, num_workers=config['train']['num_workers'], pin_memory=False)

    def gen(phase):
        epoch = config['train']['epoch_num']
        batchsize = config['train']['batchsize']
        loader = loaders[phase].__iter__()
        batch_iterator = iter(loader)
        for e in range(epoch):
            print('----',e)
            try:
                imgs, heatmaps, labels =  next(batch_iterator)
                yield {
                    'imgs': imgs,  # cropped and augmented
                    'heatmaps': heatmaps,  # based on keypoints. 0 if not in img for joint
                    'labels': labels
                }
            except StopIteration:
                batch_iterator = iter(loader)
                imgs, heatmaps, labelss = next(batch_iterator)
                yield {
                    'imgs': imgs,  # cropped and augmented
                    'heatmaps': heatmaps,  # based on keypoints. 0 if not in img for joint
                    'labels': labels
                }



    return lambda key: gen(key)
