import shutil
from os.path import dirname

import cv2
import torch
import os

from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, transforms

from data import ref
from task import cfg, update_config
from task.pose import do_test
from train import parse_command_line
from utils.group import HeatmapParser
import utils.img
import glob
import json
import re
import pandas as pd
from utils import tcUtils
from utils.misc import importNet

parser = HeatmapParser()

import argparse
from datetime import datetime
from pytz import timezone
import time
import torch
import importlib
import numpy as np
import matplotlib.pyplot as plt


def reload(config):
    """
    load or initialize model's parameters by config from config['opt'].continue_exp
    config['train']['epoch'] records the epoch num
    config['inference']['net'] is the model
    """
    opt = config['opt']

    if opt.continue_exp:
        resume = os.path.join('exp', opt.continue_exp)
        resume_file = os.path.join(resume, 'model_best.pt')
        if os.path.isfile(resume_file):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume_file, map_location='cpu')

            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            # load params
            config['inference']['net'].load_state_dict(new_state_dict)
        else:
            print("=> no checkpoint found at '{}'".format(resume))
            exit(0)

    if 'epoch' not in config['train']:
        config['train']['epoch'] = 0


def save_checkpoint(state, is_best, filename='checkpoint.pt'):
    """
    from pytorch/examples
    """
    basename = dirname(filename)
    if not os.path.exists(basename):
        os.makedirs(basename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pt')


def save(config):
    resume = os.path.join('exp', config['opt'].exp)
    if config['opt'].exp == 'pose' and config['opt'].continue_exp is not None:
        resume = os.path.join('exp', config['opt'].continue_exp)
    resume_file = os.path.join(resume, 'checkpoint.pt')

    save_checkpoint({
        'state_dict': config['inference']['net'].state_dict(),
        'optimizer': config['train']['optimizer'].state_dict(),
        'epoch': config['train']['epoch'],
    }, False, filename=resume_file)
    print('=> save checkpoint')


def init():
    """
    task.__config__ contains the variables that control the training and testing
    make_network builds a function which can do forward and backward propagation
    """
    opt = parse_command_line()
    task = importlib.import_module('task.pose')

    configs = task.__config__
    configs['opt'] = opt
    train_cfg = configs['train']
    config = configs['inference']

    update_config(cfg, opt)
    config['cfg'] = cfg
    ## creating new posenet
    PoseNet = importNet(configs['network'])
    poseNet = PoseNet(cfg,False)

    config['net'] = poseNet
    reload(configs)

    return configs

import re

re_digits = re.compile(r'(\d+)')


def embedded_numbers(s):
    pieces = re_digits.split(s)  # 切成数字和非数字
    pieces[1::2] = map(int, pieces[1::2])  # 将数字部分转成整数
    return pieces


def sort_string(lst):
    return sorted(lst, key=embedded_numbers)


def test():
    trainPath = os.getcwd()+r'/../data/testB'
    config = init()
    config['inference']['net'].eval()
    input_res = config['train']['input_res']
    output_res = config['train']['output_res']
    nstack = config['inference']['nstack']
    tans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    print('loading data...')
    tic = time.time()
    info_path = '/test_info.csv'
    try:
        info_result = pd.read_csv(trainPath + info_path)
        # result.annotation.apply(self.jsonLoads)
    except:
        info_result = read_info(trainPath)
        info_result.to_csv(trainPath + info_path, sep=',', header=True)
    try:
        with open(trainPath + "/series_map.json", "r") as f:
            study_t2_map = json.load(f)
            print("加载入文件完成...")
            pd_map = pd.DataFrame(study_t2_map)
            info_result = pd.merge(info_result, pd_map, how='right', on=['studyUid', 'seriesUid'])
    except:
        pass


    uid = info_result.groupby(['studyUid', 'seriesUid'])
    result_list = []
    print(len(uid))
    study_score = {}
    study_lang = {}
    study_result = {}

    for key, group in uid:
        group_count = group.shape[0]
        if group_count > 5:
            zIndex = int(group_count / 2)
            if group_count % 2 == 0:
                build(config, info_result, input_res, key, nstack, output_res, study_lang, study_result, study_score,
                      tans,
                      zIndex)
                build(config, info_result, input_res, key, nstack, output_res, study_lang, study_result, study_score,
                      tans,
                      zIndex + 1)
            else:
                # build(config, info_result, input_res, key, nstack, output_res, study_lang, study_result, study_score,
                #       tans,
                #       zIndex)
                build(config, info_result, input_res, key, nstack, output_res, study_lang, study_result, study_score,
                      tans,
                      zIndex + 1)
                # build(config, info_result, input_res, key, nstack, output_res, study_lang, study_result, study_score,
                #       tans,
                #       zIndex + 2)

    print(tic)
    with open("../submit/submit_"+datetime.now().strftime('%Y%m%d_%H%M%S') + ".csv", 'w', encoding='utf-8') as f:
        f.write(json.dumps([d for d in study_result.values()], ensure_ascii=False))
    print('Done {} (t={:0.2f}s)'.format(len(study_score), time.time() - tic))


def build(config, info_result, input_res, key, nstack, output_res, study_lang, study_result, study_score, tans, zIndex):
    frame_dict = {}
    frame = info_result.loc[info_result['instanceUid'] == (key[1] + '.' + str(zIndex))]
    if not frame.shape[0] == 0:
        frame_dict[frame['instanceUid'].values[0]] = frame

    paths = info_result.loc[info_result['seriesUid'] == key[1]]['dcmPath'].sort_values().values
    paths = sort_string(paths)
    frame = info_result.loc[info_result['dcmPath'] == paths[zIndex - 1]]
    if not frame.shape[0] == 0:
        frame_dict[frame['instanceUid'].values[0]] = frame

    instances = info_result.loc[info_result['seriesUid'] == key[1]]['instanceUid'].sort_values().values
    for instance in instances:
        if instance[-1] == str(zIndex):
            frame = info_result.loc[info_result['instanceUid'] == instance]
    if not frame.shape[0] == 0:
        frame_dict[frame['instanceUid'].values[0]] = frame
    for kid, instanceUid in enumerate(frame_dict):
        build_one(config, frame_dict[instanceUid], input_res, nstack, study_lang, study_result, study_score, tans, zIndex)

def build_one(config, frame, input_res, nstack, study_lang, study_result, study_score, tans, zIndex):
    frame_info = [frame['dcmPath'].values[0], frame['instanceUid'].values[0], frame['seriesUid'].values[0],
                  frame['studyUid'].values[0], ]
    path = frame_info[0]
    orig_img = tcUtils.dicom2array(path)
    input_w, input_h = orig_img.shape
    inp_img = cv2.resize(orig_img, (input_res, input_res)).astype(np.float32)
    img = tans(inp_img)
    img = img.unsqueeze(0)
    out = do_test({'imgs': img}, config)
    for o in out:
        data = {}
        data['seriesUid'] = frame_info[2]
        data['instanceUid'] = frame_info[1]
        annotations = []

        a_point = []
        conf = 0
        for oid, oo in enumerate(o[nstack]):
            p_data = {}
            if oid % 2 == 0:
                la = []
                n=int(oo[3])
                while n > 0:
                    la.append('v'+str(n % 10))
                    n = int(n / 10)

                p_data['tag'] = {'identification': ref.parts[oid], 'disc': ','.join(la)}
            else:
                p_data['tag'] = {'identification': ref.parts[oid], 'vertebra': 'v' + str(int(oo[3]))}
            p_data['coord'] = [int(oo[2] * input_h / input_res), int(oo[1] * input_w / input_res)]
            try:
                orig_img[p_data['coord'][1] - 2:p_data['coord'][1] + 2,
                p_data['coord'][0] - 2:p_data['coord'][0] + 2] = 255
            except:
                pass
            p_data['zIndex'] = zIndex
            a_point.append(p_data)
            conf += oo[0]
        a_data = {'point': a_point}
        annotations.append({"annotator": 70, 'data': a_data})
        data['annotation'] = annotations
        result = {"studyUid": frame_info[3], "version": "v0.1", "data": [data]}
        lang = inp_img.mean()
        if not study_score.get(frame_info[3]) or study_score.get(frame_info[3]) < conf:
            # and (not study_score.get(frame_info[3]) or study_score.get(frame_info[3]) < conf)
            # if study_score.get(frame_info[3]) is not None and study_score.get(frame_info[3]) > conf + 1:
            #     continue
            print(frame_info[3], frame_info[1], study_score.get(frame_info[3]), conf)
            study_result[frame_info[3]] = result
            study_lang[frame_info[3]] = lang
            study_score[frame_info[3]] = conf
            # plt.imshow(orig_img)
            # plt.title(frame_info[3] + '---' + str(conf))
            # plt.show()
        else:
            if not study_score.get(frame_info[3]):
                study_score[frame_info[3]] = 0
                study_lang[frame_info[3]] = 0
            print(frame_info[3], study_score.get(frame_info[3]), conf)


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


if __name__ == '__main__':
    test()
