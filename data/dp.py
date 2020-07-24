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
from task import cfg, update_config
from train import parse_command_line


class GenerateHeatmap():
    def __init__(self, output_res, num_parts):
        self.output_res = output_res
        self.num_parts = num_parts
        sigma = self.output_res / 64
        self.sigma = sigma
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, keypoints):
        hms = np.zeros(shape=(self.num_parts, self.output_res, self.output_res), dtype=np.float32)
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


def random_contrast(image, lower=0.5, upper=1.5):
    if random.randint(0, 1):
        # 生成随机因子
        alpha = random.uniform(lower, upper)
        image *= alpha
    return image


def random_brightness(image, delta=32):
    if random.randint(0, 1):
        delta = random.uniform(-delta, delta)
        # 图像中的每个像素加上一个随机值
        image += delta
    return image


class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, ds, size, index, is_deal, transforms, is_show=False):
        self.input_res = config['train']['input_res']
        self.output_res = config['train']['output_res']
        self.size = size
        self.num_deconvs = config['inference']['nstack']
        self.generateHeatmap = [GenerateHeatmap(int(self.output_res * (2 ** (i))), config['inference']['num_parts']) for
                                i in range(self.num_deconvs + 1)]
        self.ds = ds
        self.index = index
        self.is_deal = is_deal
        self.transforms = transforms
        self.is_show = is_show

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.loadImage(idx % self.index)

    def loadImage(self, idx):
        ds = self.ds

        orig_img,keypoints,labels = ds.getAnnots(idx)

        width, height = orig_img.shape[0:2]
        img_aug_scale = np.array([self.input_res / height, self.input_res / width])
        mat_mask_pre = utils.img.get_transform_mat([0, 0], img_aug_scale, 0)
        kpt_change_pre = utils.img.kpt_change(keypoints, mat_mask_pre)

        inp_img = cv2.resize(orig_img, (self.input_res, self.input_res))
        if self.is_deal:
            inp, kpt_affine = self.geometric_changes(inp_img, kpt_change_pre, self.input_res, self.input_res)
            inp = inp.astype(np.float32)
            # inp, kpt_affine = self.geometric_rotation_changes(inp, keypoints, kpt_affine)

            inp = random_contrast(inp, 0.5, 1)
            inp = random_brightness(inp, 32)
            inp = random_contrast(inp, 1, 1.5)
            inp = random_brightness(inp, 32)
            inp = inp.clip(0, 255)
            ## generate heatmaps on outres
            # inp = inp.astype(np.int).astype(np.float32)


        else:
            kpt_affine = kpt_change_pre
            inp = inp_img.astype(np.float32)

        mat_post = cv2.getRotationMatrix2D((0, 0), 0, self.output_res / self.input_res)
        kpt_change = utils.img.kpt_affine(kpt_affine, mat_post)

        offset = np.zeros(keypoints.shape)
        for ind, k in enumerate(kpt_change):
            offset[ind, 0] = k[0] - int(k[0])
            offset[ind, 1] = k[1] - int(k[1])
        # print(offset)
        kpt_int = kpt_change.astype(np.int)[:, [1, 0]]
        labels = np.column_stack((labels, offset, kpt_int))
        kpts = [(kpt_change * (2 ** i)).astype(np.int) for i in range(self.num_deconvs + 1)]
        heatmaps = [self.generateHeatmap[i](kpts[i]) for i in range(self.num_deconvs + 1)]
        masks = []
        for heatmap in heatmaps:
            mask = np.zeros(heatmap.shape, dtype=np.float32)
            mask[heatmap > 0] = 1
            masks.append(mask)
        if self.is_show:
            self.show(heatmaps, inp, inp_img, kpts, kpt_change_pre)
        return self.transforms(inp[np.newaxis, :, :]), heatmaps, np.array(labels).astype(np.float32),masks

    def geometric_rotation_changes(self, inp_img, keypoints, kpt_change_pre):
        scale = random.uniform(0.75, 1.25)
        aug_rot = random.uniform(-20, 20)
        center = kpt_change_pre[random.randint(0, keypoints.shape[0] - 1)]
        center = [center[0] * random.uniform(0.75, 1.25), center[1] * random.uniform(0.75, 1.25)]
        mat = cv2.getRotationMatrix2D((center[1], center[0]), aug_rot, scale)
        img_scale_poss = [random.uniform(0.75, 1.25), random.uniform(0.75, 1.25)]
        resize = [int(self.input_res * img_scale_poss[0]), int(self.input_res * img_scale_poss[1])]
        inp = cv2.warpAffine(inp_img, mat, (resize[0], resize[1])).astype(np.float32)
        inp = cv2.resize(inp, (self.input_res, self.input_res))
        kpt_affine = utils.img.kpt_affine(kpt_change_pre, mat)
        mat_mask_poss = utils.img.get_transform_mat([0, 0],
                                                    [self.input_res / resize[0], self.input_res / resize[1]], 0)
        kpt_affine = utils.img.kpt_change(kpt_affine, mat_mask_poss)
        return inp, kpt_affine

    def show(self, heatmaps, inp, inp_img, kpt_change, kpt_change_pre):
        for ind, ks in enumerate(kpt_change):
            show_inp = cv2.resize(inp, (self.output_res * (2 ** (ind)), self.output_res * (2 ** (ind))))
            heatmaps_show = np.zeros([self.output_res * (2 ** (ind)), self.output_res * (2 ** (ind))])
            for jnd, k in enumerate(ks):
                if k[0] <= 0 or k[1] <= 0 or k[0] >= self.output_res * (2 ** (ind)) or k[1] >= self.output_res * (
                        2 ** (ind)):
                    continue
                show_inp[k[1], k[0]] = 255
                heatmaps_show += heatmaps[ind][jnd, :, :]
            plt.imshow(heatmaps_show)
            plt.show()
            plt.imshow(show_inp)
            plt.show()
        plt.imshow(inp)
        plt.show()

        for k in kpt_change_pre:
            inp_img[int(k[1]), int(k[0])] = 255
        plt.imshow(inp_img)
        plt.show()

    def geometric_changes(self,image,kpt_change_pre,width,height):
        pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
        rand = np.random.randint(-30,30,size=(3,2))
        pts2 = pts1+rand
        M = cv2.getAffineTransform(pts1, pts2.astype(np.float32))
        image_0 = cv2.warpAffine(image, M, (width, height))
        kpt_affine = utils.img.kpt_affine(kpt_change_pre, M)
        return image_0,kpt_affine

def init(config):
    batchsize = config['train']['batchsize']
    current_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_path)
    train = config['inference']['train_num_eval']

    annot_path = r'/annotation.json'
    train_data_dir = r'/home/dwxt/project/dcm/train'
    valid_data_dir = r'/home/dwxt/project/dcm/valid'
    info_name = r'/info.csv'
    iters = config['train']['train_iters']
    batch_size = config['train']['batchsize']
    tans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    train_db = Dataset(config, ds.Lumbar(train_data_dir, annot_path, info_name), iters * batch_size, 150, True, tans)

    valid_train_db = Dataset(config, ds.Lumbar(train_data_dir, annot_path, info_name), config['train']['valid_train_iters'] * batch_size, 150, True, tans)

    train_valid_db = Dataset(config, ds.Lumbar(valid_data_dir, annot_path, info_name),
                             config['train']['train_valid_iters'] * batch_size, 51, True, tans)
    valid_db = Dataset(config, ds.Lumbar(valid_data_dir, annot_path, info_name),
                       config['train']['valid_iters'] * batch_size, 51, False, tans)

    dataset = {'train': train_db, 'train_valid': train_valid_db, 'valid': valid_db,'valid_train':valid_train_db}

    loaders = {}
    for key in dataset:
        loaders[key] = torch.utils.data.DataLoader(dataset[key], batch_size=batchsize, shuffle=True,
                                                   num_workers=config['train']['num_workers'], pin_memory=False)

    return loaders


if __name__ == '__main__':
    import task.pose

    config = task.pose.__config__
    annot_path = r'/annotation.json'
    train_data_dir = r'E:\data\lumbar_train51\train'
    info_name = r'/info.csv'
    size = config['train']['epoch_num'] * config['train']['data_num']
    tans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    train_db = Dataset(config, ds.Lumbar(train_data_dir, annot_path, info_name), size, 150, True, tans, True)

    for i in range(51):
        train_db.loadImage(i)
