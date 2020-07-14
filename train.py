import os
import shutil

import tqdm
from os.path import dirname

import torch.backends.cudnn as cudnn

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
                        required=True,
                        default="",
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
        resume_file = os.path.join(resume, 'model_best.pt')
        if os.path.isfile(resume_file):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume_file)

            config['inference']['net'].load_state_dict(checkpoint['state_dict'], False)
            config['train']['optimizer'].load_state_dict(checkpoint['optimizer'])
            config['train']['epoch'] = checkpoint['epoch']
            # config['train']['epoch'] = 0
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


def train(train_func, data_func, config, post_epoch=None):
    save_loss = 1e10
    while True:
        fails = 0
        print('epoch: ', config['train']['epoch'])
        if 'epoch_num' in config['train']:
            if config['train']['epoch'] > config['train']['epoch_num']:
                break

        all_loss = 0
        for phase in ['train', 'valid']:
            num_step = config['train']['{}_iters'.format(phase)]
            generator = data_func(phase)
            print('start', phase, config['opt'].exp)

            show_range = range(num_step)
            show_range = tqdm.tqdm(show_range, total=num_step, ascii=True)
            batch_id = num_step * config['train']['epoch']
            if batch_id > config['opt'].max_iters * 1000:
                return
            losses = []
            for i in show_range:
                datas = next(generator)
                loss = train_func(batch_id + i, config, phase, **datas)
                losses.append(loss)
                # print('epoch: {}; step: {}; loss: {}'.format(config['train']['epoch'],batch_id+i,loss))
        config['train']['epoch'] += 1
        mean_loss = np.mean(np.array(losses))
        print('valid loss:',save_loss,mean_loss)
        if mean_loss < save_loss:
            save_loss = mean_loss
            save(config, True)
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

    func = task.make_network(config)
    reload(config)
    return func, config


def main():
    func, config = init()
    data_func = config['data_provider'].init(config)
    train(func, data_func, config)
    print(datetime.now(timezone('EST')))


if __name__ == '__main__':
    main()
