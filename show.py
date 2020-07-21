import shutil
from os.path import dirname

import cv2
import torch
import os

from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor

from data import ref
from utils.group import HeatmapParser
import utils.img
import glob
import json
import re
import pandas as pd
from utils import tcUtils

parser = HeatmapParser()

import argparse
from datetime import datetime
from pytz import timezone
import time
import torch
import importlib
import numpy as np
import matplotlib.pyplot as plt




def read_info(trainPath):
    dcm_paths = glob.glob(os.path.join(trainPath, "**", "**.dcm"))
    # 'studyUid','seriesUid','instanceUid'
    tag_list = ['0020|000d', '0020|000e', '0008|0018']
    dcm_info = pd.DataFrame(columns=('dcmPath', 'studyUid', 'seriesUid', 'instanceUid'))
    for dcm_path in dcm_paths:
        try:
            studyUid, seriesUid, instanceUid = tcUtils.dicom_metainfo(dcm_path, tag_list)
            row = pd.Series(
                {'dcmPath': dcm_path, 'studyUid': studyUid, 'seriesUid': seriesUid, 'instanceUid': instanceUid})
            dcm_info = dcm_info.append(row, ignore_index=True)
        except:
            continue
    return dcm_info


trainPath = r'/home/dwxt/project/dcm/test'
info_path = '/test_info.csv'
try:
    info_result = pd.read_csv(trainPath + info_path)
    # result.annotation.apply(self.jsonLoads)
except:
    info_result = read_info(trainPath)
    info_result.to_csv(trainPath + info_path, sep=',', header=True)
with open('data-1595208936.575084.json', 'r', encoding='utf-8') as f:
    study_result = json.load(f)
for i, v in enumerate(study_result):
    for data_ in v['data']:
        instanceUid = data_['instanceUid']
        print(instanceUid)
        loc_result = info_result.loc[info_result['instanceUid'] == instanceUid]
        orig_img = tcUtils.dicom2array(loc_result['dcmPath'].values[0])
        points = data_['annotation'][0]['data']['point']
        for p in points:
            print(p['coord'][1], p['coord'][0])
            if p['coord'][1]>orig_img.shape[0] or p['coord'][0]>orig_img.shape[1]:
                continue
            orig_img[p['coord'][1]-2:p['coord'][1]+2, p['coord'][0]-2:p['coord'][0]+2] = 255
        plt.imshow(orig_img)
        plt.title(instanceUid)
        plt.show()