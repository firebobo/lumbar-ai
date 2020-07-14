import random

import cv2
import sys
import os
import torch
import numpy as np
from PIL import Image
import torch.utils.data
from torchvision.transforms import Compose, Resize, ToTensor, transforms
import matplotlib.pyplot as plt
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
                if x <= 0 or y <= 0 or x >= self.output_res or y >= self.output_res:
                    continue
                ul = int(x - 3 * sigma - 1), int(y - 3 * sigma - 1)
                br = int(x + 3 * sigma + 2), int(y + 3 * sigma + 2)

                c, d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                a, b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                cc, dd = max(0, ul[0]), min(br[0], self.output_res)
                aa, bb = max(0, ul[1]), min(br[1], self.output_res)
                hms[idx, aa:bb, cc:dd] = np.maximum(hms[idx, aa:bb, cc:dd], self.g[a:b, c:d])
        return hms


class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, ds,size, index,is_deal):
        self.input_res = config['train']['input_res']
        self.output_res = config['train']['output_res']
        self.size = size
        self.num_deconvs = config['cif'].MODEL.EXTRA.DECONV.NUM_DECONVS
        self.generateHeatmap = [GenerateHeatmap(self.output_res*(i+1), config['inference']['num_parts']) for i in (self.num_deconvs+1)]
        self.ds = ds
        self.index = index
        self.is_deal = is_deal


    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        index = int(random.random() * self.index + idx) % self.index
        return self.loadImage(index)

    def loadImage(self, idx):
        ds = self.ds

        orig_img = ds.get_img(idx)
        keypoints = ds.get_kps(idx)
        labels = ds.get_labels(idx)

        width,height = orig_img.shape[0:2]


        img_aug_scale = np.array([self.input_res/height,self.input_res/width])
        mat_mask_pre = utils.img.get_transform_mat([0,0], img_aug_scale, 0)
        kpt_change_pre = utils.img.kpt_change(keypoints, mat_mask_pre)


        inp_img = cv2.resize(orig_img, (self.input_res, self.input_res))
        if self.is_deal:
            center = kpt_change_pre[int(np.random.random() * keypoints.shape[0])]
            scale = (np.random.random()-0.5)* 0.4 + 1

            aug_rot = (np.random.random()-.5) * 60

            # diff = (np.random.random((self.input_res, self.input_res)) -.5) * 20
            # inp_img = inp_img+diff
            mat = cv2.getRotationMatrix2D((center[1], center[0]), aug_rot, scale)

            inp = cv2.warpAffine(inp_img, mat, (self.input_res, self.input_res)).astype(np.float32)
            ## generate heatmaps on outres

            kpt_affine = utils.img.kpt_affine(kpt_change_pre, mat)
        else:
            kpt_affine = kpt_change_pre
            inp = inp_img.astype(np.float32)

        mat_post = cv2.getRotationMatrix2D((0,0), 0, self.output_res/self.input_res)
        kpt_change = utils.img.kpt_affine(kpt_affine, mat_post)

        # offset = np.zeros(keypoints.shape)
        # for ind,k in enumerate(kpt_change):
        #     offset[ind,0]=k[0] - int(k[0])
        #     offset[ind,1]=k[1] - int(k[1])
        # print(offset)
        kpt_int = kpt_change.astype(np.int)[:,[1,0]]
        # labels = np.column_stack((labels, kpt_change))
        heatmaps = [self.generateHeatmap[i](kpt_change**(i+1)) for i in (self.num_deconvs+1)]

        # self.show(heatmaps, inp, inp_img, kpt_int, kpt_change_pre)
        return inp[np.newaxis,:,:], heatmaps,np.array(labels).astype(np.float32)

    def show(self, heatmaps, inp, inp_img, kpt_change, kpt_change_pre):
        show_inp = cv2.resize(inp, (self.output_res, self.output_res))
        heatmaps_show = np.zeros([self.output_res, self.output_res])
        for ind, k in enumerate(kpt_change):
            if k[0] <= 0 or k[1] <= 0 or k[0] >= self.output_res or k[1] >= self.output_res:
                continue
            show_inp[k[0], k[1]] = 255
            heatmaps_show += heatmaps[ind, :, :]
        plt.imshow(heatmaps_show)
        plt.show()
        plt.imshow(inp)
        plt.show()
        plt.imshow(show_inp)
        plt.show()
        for k in kpt_change_pre:
            inp_img[int(k[1]), int(k[0])] = 255
        plt.imshow(inp_img)
        # # plt.imshow(show_inp)
        plt.show()

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

    train_db = Dataset(config, ds.Lumbar(train_data_dir, annot_path, info_name), size, 150,True)
    valid_db = Dataset(config, ds.Lumbar(valid_data_dir, annot_path, info_name), size, 51,True)

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

if __name__ == '__main__':
    import task.pose
    config = task.pose.__config__
    annot_path = r'/annotation.json'
    train_data_dir = r'E:\data\lumbar_train51\train'
    info_name = r'/info.csv'
    size = config['train']['epoch_num'] * config['train']['data_num']

    train_db = Dataset(config, ds.Lumbar(train_data_dir, annot_path, info_name), size, 150,True)
    train_db.loadImage(10)
