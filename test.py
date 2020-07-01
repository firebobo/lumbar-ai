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
trainPath = r'/home/dwxt/project/dcm/test'
import argparse
from datetime import datetime
from pytz import timezone
import time
import torch
import importlib


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--continue_exp', type=str, help='continue exp')
    parser.add_argument('-e', '--exp', type=str, default='pose', help='experiments name')
    parser.add_argument('-m', '--max_iters', type=int, default=250, help='max number of iterations (thousands)')
    args = parser.parse_args()
    return args


def reload(config):
    """
    load or initialize model's parameters by config from config['opt'].continue_exp
    config['train']['epoch'] records the epoch num
    config['inference']['net'] is the model
    """
    opt = config['opt']

    if opt.continue_exp:
        resume = os.path.join('exp', opt.continue_exp)
        resume_file = os.path.join(resume, 'checkpoint.pt')
        if os.path.isfile(resume_file):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume_file)

            config['inference']['net'].load_state_dict(checkpoint['state_dict'])
            config['train']['optimizer'].load_state_dict(checkpoint['optimizer'])
            config['train']['epoch'] = checkpoint['epoch']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
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
    exp_path = os.path.join('exp', opt.exp)

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')

    config = task.__config__
    try:
        os.makedirs(exp_path)
    except FileExistsError:
        pass

    config['opt'] = opt
    config['data_provider'] = importlib.import_module(config['data_provider'])

    func = task.make_network(config)
    reload(config)
    return func, config


def test():
    run_func, config = init()
    input_res = config['train']['input_res']
    output_res = config['train']['output_res']
    nstack = config['inference']['nstack']
    transforms = Compose([Resize((input_res, input_res)), ToTensor()])
    print('loading data...')
    tic = time.time()
    info_path = '/test_info.csv'
    try:
        info_result = pd.read_csv(trainPath + info_path)
        # result.annotation.apply(self.jsonLoads)
    except:
        info_result = read_info()
        info_result.to_csv(trainPath + info_path, sep=',', header=True)
    uid = info_result.groupby(['studyUid', 'seriesUid'])
    result_list = []
    for key,group in uid:
        group_count = group.shape[0]
        if group_count >5:
            frame = info_result.loc[info_result['instanceUid']==(key[1]+'.'+str(int(group_count/2)))]
            if frame.shape[0]==0:
                continue
            frame_info = [frame['dcmPath'].values[0],frame['instanceUid'].values[0],frame['seriesUid'].values[0],frame['studyUid'].values[0],]
            path = frame_info[0]
            orig_img=tcUtils.dicom2array(path)
            input_r = orig_img.shape[0]
            img = transforms(Image.fromarray(orig_img)).unsqueeze(0)
            out = run_func(key,config,"inference",**{'imgs':img})
            for o in out:
                data = {}
                data['seriesUid'] = frame_info[2]
                data['instanceUid'] = frame_info[1]
                annotations = []

                a_point = []
                conf = 0
                for oid,oo in enumerate(o[nstack-1]):
                    p_data={}
                    p_data['coord']= [int(oo[1]*input_r/output_res),int(oo[2]*input_r/output_res)]
                    p_data['tag'] = {'identification':ref.parts[oid],'disc':'v'+str(int(oo[3]))}
                    a_point.append(p_data)
                    conf += oo[0]
                a_data={'point':a_point}
                annotations.append(a_data)
                data['annotations'] = annotations
                result = {"studyUid":frame_info[3],"data":data}
                print(conf)
                if(conf>3):
                    result_list.append(result)
    print(result_list)
    print('Done (t={:0.2f}s)'.format(time.time() - tic))




def read_info():
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
