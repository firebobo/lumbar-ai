import numpy as np
import os 
import time
import glob
import pandas as pd
from utils import tcUtils

def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

annot_path = r'/home/dwxt/project/dcm/lumbar_train150_annotation.json'
img_dir = r'/home/dwxt/project/dcm/lumbar_train150'

assert os.path.exists(img_dir)
mpii, num_examples_train, num_examples_val = None, None, None

import cv2

class Lumbar:
    def __init__(self):
        print('loading data...')
        tic = time.time()

        # studyUid,seriesUid,instanceUid,annotation
        annotation_info = pd.DataFrame(columns=('studyUid', 'seriesUid', 'instanceUid', 'annotation'))
        json_df = pd.read_json(annot_path)
        for idx in json_df.index:
            studyUid = json_df.loc[idx, "studyUid"]
            seriesUid = json_df.loc[idx, "data"][0]['seriesUid']
            instanceUid = json_df.loc[idx, "data"][0]['instanceUid']
            annotation = json_df.loc[idx, "data"][0]['annotation'][0]['data']['point']
            row = pd.Series(
                {'studyUid': studyUid, 'seriesUid': seriesUid, 'instanceUid': instanceUid, 'annotation': annotation})
            annotation_info = annotation_info.append(row, ignore_index=True)

        dcm_paths = glob.glob(os.path.join(img_dir, "**", "**.dcm"))
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

        result = pd.merge(annotation_info, dcm_info, on=['studyUid', 'seriesUid', 'instanceUid'])
        self.info = result

        
        print('Done (t={:0.2f}s)'.format(time.time()- tic))

    def getAnnots(self,idx):
        row = self.info.rows[idx]
        points = row['annotation']
        kps = []
        for p in points:
            kps.append(p['coord'])
        return row['annotation']

    def getLength(self):
        return len(self.t_center), len(self.v_center)

    def get_img(self,idx):
        return tcUtils.dicom2array(self.get_path(idx))

    def get_path(self,idx):
        row = self.info.rows[idx]
        return row['dcmPath']

    def get_kps(self,idx):
        row = self.info.rows[idx]
        points = row['annotation']
        kps = {}
        for p in points:
            kps[p['tag']['identification']] = p['tag']['coord']
        key_points = []
        for part in parts:
            key_points.append(kps[part])

        return key_points

    def get_labels(self,idx):
        row = self.info.rows[idx]
        points = row['annotation']
        lbs = {}
        for p in points:
            lbs[p['tag']['identification']] = p['tag']['disc']

        labels = []

        for part in parts:
            lab = np.zeros(7, dtype=int)
            if "-" in lbs[part]:
                lab[pair_labels[1][lbs[part]]-1] = 1
            else:
                lab[pair_labels[0][lbs[part]] - 1] = 1

    
# Part reference
parts = ['T12-L1','L1', 'L1-L2', 'L2','L2-L3', 'L3', 'L3-L4',  'L4','L4-L5','L5', 'L5-S1']


pair_labels = [{'v1':1, 'v2':2}, {'v1':3, 'v2':4, 'v3':5, 'v4':6, 'v5':7}]
