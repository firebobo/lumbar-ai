import random

import numpy as np
import os 
import time
import glob
import json
import re
import pandas as pd
from utils import tcUtils

def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class Lumbar:
    def jsonLoads(self,strs):
        '''strs：传进来的json数据
           key：字典的键
        '''
        strs = re.sub("'", '"', strs)  # 单引号换成双引号，下文解释
        strs = re.sub(r'\\', '"', strs)  # 单引号换成双引号，下文解释
        dict_ = json.loads(strs)
        return dict_  # 原地代替原来的json数据，这里使用列表推导
    def __init__(self,data_dir,annot_path,info_path):
        self.data_dir = data_dir
        self.annot_path = annot_path
        self.info_path = info_path
        self.file_info_name = '/file_info.csv'

        print('loading data...')
        tic = time.time()
        try:
            result = pd.read_csv(self.data_dir + self.info_path)
            file_info = pd.read_csv(self.data_dir + self.file_info_name)

            # result.annotation.apply(self.jsonLoads)
            self.info = result
            self.file_info = file_info
        except:
            result,path_info = self.read_info()
            result.to_csv(self.data_dir + self.info_path,sep=',',header=True)
            path_info.to_csv(self.data_dir + self.file_info_name,sep=',',header=True)

            self.info = result
            self.file_info = path_info



        print('Done (t={:0.2f}s)'.format(time.time()- tic))

    def read_info(self):
        # studyUid,seriesUid,instanceUid,annotation
        annotation_info = pd.DataFrame(columns=('studyUid', 'seriesUid', 'instanceUid', 'annotation'))
        json_df = pd.read_json(self.data_dir+self.annot_path)
        for idx in json_df.index:
            studyUid = json_df.loc[idx, "studyUid"]
            seriesUid = json_df.loc[idx, "data"][0]['seriesUid']
            instanceUid = json_df.loc[idx, "data"][0]['instanceUid']
            annotation = json_df.loc[idx, "data"][0]['annotation'][0]['data']['point']
            row = pd.Series(
                {'studyUid': studyUid, 'seriesUid': seriesUid, 'instanceUid': instanceUid, 'annotation': annotation})
            annotation_info = annotation_info.append(row, ignore_index=True)
        dcm_paths = glob.glob(os.path.join(self.data_dir, "**", "**.dcm"))
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
        return result,dcm_info

    def getAnnots(self,idx):
        row = self.get_row(idx)
        points = row['annotation']
        if type(points) is str:
            points = self.jsonLoads(points)
        lbs = {}
        for p in points:
            # print(p)
            disc = None
            if p['tag'].get('vertebra'):
                disc = p['tag'].get('vertebra')
            else:
                disc = p['tag'].get('disc')
            lbs[p['tag']['identification']] = disc
        labels = np.zeros((len(parts), 7))
        for idx, part in enumerate(parts):
            if lbs.get(part):
                if "-" in part:
                    for par in lbs[part].split(','):
                        labels[idx, pair_labels[1][par] - 1] = 1
                else:
                    labels[idx, pair_labels[0][lbs[part]] - 1] = 1
        if type(points) is str:
            points = self.jsonLoads(points)
        kps = {}
        for p in points:
            kps[p['tag']['identification']] = p['coord']
        key_points = np.zeros((len(parts),2))
        for idx,part in enumerate(parts):
            if kps.get(part):
                key_points[idx] = kps[part]

        img = tcUtils.dicom2array(row['dcmPath'])

        return img,key_points,labels

    def getLength(self):
        return len(self.t_center), len(self.v_center)

    def get_img(self,idx):
        return tcUtils.dicom2array(self.get_path(idx))

    def get_path(self,idx):
        row = self.get_row(idx)
        return row['dcmPath']

    def get_row(self, idx):
        row = self.info.loc[idx]
        instanceUid = row['instanceUid']
        split = instanceUid.split('.')
        str_id = int(split[-1]) + random.randint(-1, 1)
        split[-1] = str(str_id)
        temp = '.'.join(split)
        temp_row = self.file_info.loc[self.file_info['instanceUid']==temp]
        if temp_row.empty:
            return row
        if not temp_row.get('annotation'):
            row = row.copy()
            row['dcmPath'] = temp_row['dcmPath'].values[0]
        return row

    def get_kps(self,idx):
        row = self.get_row(idx)
        points = row['annotation']
        if type(points) is str:
            points = self.jsonLoads(points)
        kps = {}
        for p in points:
            kps[p['tag']['identification']] = p['coord']
        key_points = np.zeros((len(parts),2))
        for idx,part in enumerate(parts):
            if kps.get(part):
                key_points[idx] = kps[part]

        return key_points

    def get_labels(self,idx):
        row = self.get_row(idx)
        points = row['annotation']
        if type(points) is str:
            points = self.jsonLoads(points)
        lbs = {}
        for p in points:
            # print(p)
            disc = None
            if p['tag'].get('vertebra'):
                disc = p['tag'].get('vertebra')
            else:
                disc = p['tag'].get('disc')
            lbs[p['tag']['identification']] = disc
        labels = np.zeros((len(parts),7))
        for idx,part in enumerate(parts):
            if lbs.get(part):
                if "-" in part:
                    for par in lbs[part].split(','):
                        labels[idx,pair_labels[1][par]-1] = 1
                else:
                    labels[idx,pair_labels[0][lbs[part]] - 1] = 1
        return labels

    
# Part reference
parts = ['T12-L1','L1', 'L1-L2', 'L2','L2-L3', 'L3', 'L3-L4',  'L4','L4-L5','L5', 'L5-S1']


pair_labels = [{'v1':1, 'v2':2}, {'v1':3, 'v2':4, 'v3':5, 'v4':6, 'v5':7}]
