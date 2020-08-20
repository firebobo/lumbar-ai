import os
import shutil
import sys
import data.dp
import data.ref as ds

from torchvision.transforms import transforms
from tqdm import tqdm
from os.path import dirname

import torch.backends.cudnn as cudnn
import torch.utils.data

from data.dp import Dataset
from task.pose import do_train, do_valid

cudnn.benchmark = True
cudnn.enabled = True

import torch
import numpy as np
import importlib
import argparse
import shutil
import numpy as np
from datetime import datetime
from pytz import timezone
from task import cfg,update_config


def parse_command_line():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument('-c', '--continue_exp', type=str, help='continue exp')
    parser.add_argument('-e', '--exp', type=str, default='pose', help='experiments name')
    parser.add_argument('-m', '--max_iters', type=int, default=250, help='max number of iterations (thousands)')

    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="config/w48_640_adam_lr1e-3.yaml",
                        type=str)

    parser.add_argument('--opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

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
        # resume_file = os.path.join(resume, 'checkpoint.pt')
        resume_file = os.path.join(resume, 'model_best.pt')
        if os.path.isfile(resume_file):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume_file)

            config['inference']['net'].load_state_dict(checkpoint['state_dict'], False)
            # config['train']['optimizer'].load_state_dict(checkpoint['optimizer'])
            # config['train']['epoch'] = checkpoint['epoch']
            config['train']['epoch'] = 0
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))

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
        shutil.copyfile(filename, basename + '/model_best.pt')



def save(config, is_best=False):
    resume = os.path.join('exp', config['opt'].exp)
    if config['opt'].exp == 'pose' and config['opt'].continue_exp is not None:
        resume = os.path.join('exp', config['opt'].continue_exp)
    resume_file = os.path.join(resume, 'checkpoint.pt')

    save_checkpoint({
        'state_dict': config['inference']['net'].state_dict(),
        'optimizer': config['train']['optimizer'].state_dict(),
        'epoch': config['train']['epoch'],
    }, is_best, filename=resume_file)
    print('=> save checkpoint')


def train(data_loaders, config, post_epoch=None):
    save_correct = 0
    net = config['inference']['net']
    config['inference']['net'] = net.train()
    while True:
        fails = 0
        tqdm.write('Epoch {}/{}'.format(config['train']['epoch'], config['train']['epoch_num'] - 1))
        if 'epoch_num' in config['train']:
            if config['train']['epoch'] > config['train']['epoch_num']:
                break
        # do_train(config['train']['epoch'], config, data_loaders['train_valid'])
        do_train(config['train']['epoch'], config, data_loaders['train'])
        # train_mean_correct = do_valid(config['train']['epoch'], config, data_loaders['valid_train'])
        valid_mean_correct = do_valid(config['train']['epoch'], config, data_loaders['valid'])
        # mean_correct = (train_mean_correct+valid_mean_correct)/2
        mean_correct = valid_mean_correct
        train_mean_loss = 0
        config['train']['epoch'] += 1
        config['train']['scheduler'].step()
        print('valid loss:', save_correct, mean_correct)
        if mean_correct > save_correct:
            save_correct = mean_correct
            save(config, True)
            save(config)
        else:
            save(config)




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

    update_config(cfg, opt)
    config['cfg'] = cfg
    try:
        os.makedirs(exp_path)
    except FileExistsError:
        pass

    config['opt'] = opt
    config['data_provider'] = importlib.import_module(config['data_provider'])

    task.make_network(config)
    reload(config)
    return config


def data_init(config):
    batchsize = config['train']['batchsize']
    current_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_path)

    annot_path = r'/annotation.json'
    train_data_dir = os.getcwd()+r'/../data/train'
    valid_data_dir = os.getcwd()+r'/../data/valid'
    info_name = r'/info.csv'
    iters = config['train']['train_iters']
    batch_size = config['train']['batchsize']
    tans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    train_db = Dataset(config, ds.Lumbar(train_data_dir, annot_path, info_name), iters * batch_size, 150, True, tans)

    valid_train_db = Dataset(config, ds.Lumbar(train_data_dir, annot_path, info_name),
                             config['train']['valid_train_iters'] * batch_size, 150, False, tans)

    train_valid_db = Dataset(config, ds.Lumbar(valid_data_dir, annot_path, info_name),
                             config['train']['train_valid_iters'] * batch_size, 51, True, tans)
    valid_db = Dataset(config, ds.Lumbar(valid_data_dir, annot_path, info_name),
                       config['train']['valid_iters'] * batch_size, 51, False, tans)

    dataset = {'train': train_db, 'train_valid': train_valid_db, 'valid': valid_db, 'valid_train': valid_train_db}

    loaders = {}
    for key in dataset:
        loaders[key] = torch.utils.data.DataLoader(dataset[key], batch_size=batchsize, shuffle=True,
                                                   num_workers=config['train']['num_workers'], pin_memory=False)

    return loaders


def main():
    config = init()
    data_loaders = data_init(config)
    train(data_loaders, config)
    print(datetime.now(timezone('EST')))

if __name__ == '__main__':
    main()
