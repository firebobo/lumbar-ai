"""
__config__ contains the options for training and testing
Basically all of the variables related to training are put in __config__['train'] 
"""
import torch
import numpy as np
from torch import nn
import os
from torch.nn import DataParallel
from torch.optim import lr_scheduler
from tqdm import tqdm

from task.loss import KeypointLoss
from utils.misc import make_input, make_output, importNet

__config__ = {
    'data_provider': 'data.dp',
    'network': 'models.pose_higher_hrnet.get_pose_net',
    'inference': {
        'nstack': 2,
        'inp_dim': 256,
        'oup_dim': 11,
        'num_parts': 11,
        'num_class': 9,
        'increase': 0,
        'keys': ['imgs'],
        'num_eval': 2958,  ## number of val examples used. entire set is 2958
        'train_num_eval': 150,  ## number of train examples tested at test time
    },

    'train': {
        'batchsize': 24,
        'input_res': 256,
        'output_res': 64,
        'epoch_num': 300,
        'data_num': 150,
        'train_iters': 1500,
        'valid_iters': 10,
        'learning_rate': 1e-3,
        'max_num_people': 1,
        'loss': [1,1],
        'stack_loss': [1, 1, 1],
        'decay_iters': 50,
        'decay_lr': 0.1,
        'num_workers': 2,
        'use_data_loader': True,
        'train_num_eval': 150,
        'valid_num_eval': 51
    },
}


def build_targets(preds):
    # print(gt.shape)
    nstask = len(preds)
    size = preds[0].shape
    targets = np.zeros([size[0], nstask, int(size[1]/2), 4])

    m = size[2]
    n = size[3]
    for idx in range(nstask):
        heatmap, labelmap = preds[idx][:,int(size[1]/2):],preds[idx][:,:int(size[1]/2)]
        for jdx, gg in enumerate(heatmap):
            for kdx in range(int(size[1]/2)):
                a = heatmap[jdx, kdx]
                index = int(a.argmax())
                x = int(index / n)
                y = index % n
                targets[idx, jdx, kdx, 0] = a[x, y]
                targets[idx, jdx, kdx, 1:3] = [x, y]
                if kdx % 2 == 0:
                    y_ = labelmap[jdx, 2:7, x, y]
                    ind = y_.argmax()
                else:
                    y_ = labelmap[jdx, :2, x, y]
                    ind = y_.argmax()
                targets[idx, jdx, kdx, 3] = ind + 1

        return targets  ## l of dim bsize


def make_network(configs):
    train_cfg = configs['train']
    config = configs['inference']

    ## creating new posenet
    PoseNet = importNet(configs['network'])
    poseNet = PoseNet(configs["cfg"],True)
    forward_net = DataParallel(poseNet.cuda())
    config['net'] = forward_net
    config['lossLayers'] = KeypointLoss(configs['inference']['num_parts'], 2,
                                        configs['inference']['num_class'])
    ## optimizer, experiment setup
    train_cfg['optimizer'] = torch.optim.Adam(filter(lambda p: p.requires_grad, config['net'].parameters()), train_cfg['learning_rate'])
    train_cfg['scheduler'] = lr_scheduler.StepLR(train_cfg['optimizer'], step_size=train_cfg['decay_iters'],
                                                 gamma=train_cfg['decay_lr'])  # 更新学习率的策略：  每隔step_size个epoch就将学习率降为原来的gamma倍。
    exp_path = os.path.join('exp', configs['opt'].exp)
    if configs['opt'].exp == 'pose' and configs['opt'].continue_exp is not None:
        exp_path = os.path.join('exp', configs['opt'].continue_exp)
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)
    logger = open(os.path.join(exp_path, 'log'), 'a+')
    config['logger'] = logger


def do_train(epoch, config, loader):
    logger = config['inference']['logger']
    net = config['inference']['net']

    batch_idx = 0
    for inputs in tqdm(loader):
        for i, input in enumerate(inputs):
            try:
                if type(inputs[i]) is list:
                    for ind, inp in enumerate(inputs[i]):
                        inputs[i][ind] = make_input(inp)
                else:
                    inputs[i] = make_input(inputs[i])
            except:
                pass  # for last input, which is a string (id_)

        combined_preds = net(inputs[0])
        combined_loss, labels_loss = config['inference']["lossLayers"](combined_preds,
                                                                       **{'heatmaps': inputs[1], 'labels': inputs[2]})
        num_loss = config['train']['loss']

        # losses = [all_loss[idx].cpu()*i[1] for idx, i in enumerate(config['train']['loss'])]
        heatmap_loss = torch.sum(combined_loss.cpu().mul(torch.Tensor(config['train']['stack_loss'])))
        label_loss = torch.sum(labels_loss.cpu().mul(torch.Tensor(config['train']['stack_loss'])))
        toprint = '\n{} {}: '.format(epoch, batch_idx)

        if batch_idx % 100 == 0:
            toprint += ' \n{}'.format(str(combined_loss.cpu()))
            toprint += ' \n{}'.format(str(labels_loss.cpu()))
        else:
            toprint += ' {},{}'.format(heatmap_loss, label_loss)

        logger.write(toprint)
        logger.flush()
        optimizer = config['train']['optimizer']
        optimizer.zero_grad()
        loss = num_loss[0] * heatmap_loss + num_loss[1] * label_loss
        # loss = heatmap_loss
        loss.backward()
        optimizer.step()
        batch_idx += 1


def do_valid(epoch, config, loader):
    logger = config['inference']['logger']
    net = config['inference']['net']
    run_loss = 0
    batch_idx = 0
    with torch.no_grad():

        for inputs in tqdm(loader):
            for i, input in enumerate(inputs):
                if type(inputs[i]) is list:
                    for ind, inp in enumerate(inputs[i]):
                        inputs[i][ind] = make_input(inp)
                else:
                    inputs[i] = make_input(inputs[i])

            combined_preds = net(inputs[0])
            combined_loss, labels_loss = config['inference']["lossLayers"](combined_preds, **{'heatmaps': inputs[1],
                                                                                              'labels': inputs[2]})
            loss = labels_loss[:, -1].sum() + combined_loss[:, -1].sum()

            toprint = '\n{} {}: '.format(epoch, batch_idx)
            heatmap_loss = torch.sum(combined_loss.cpu().mul(torch.Tensor(config['train']['stack_loss'])))
            label_loss = torch.sum(labels_loss.cpu().mul(torch.Tensor(config['train']['stack_loss'])))
            if batch_idx % 100 == 0:
                toprint += ' \n{}'.format(str(combined_loss.cpu()))
                toprint += ' \n{}'.format(str(labels_loss.cpu()))
            else:
                toprint += ' {},{}'.format(heatmap_loss, label_loss)
            logger.write(toprint)
            logger.flush()
            batch_idx += 1
            run_loss += loss.float().item()
    return run_loss/batch_idx

def do_test(inputs, config):
    net = config['inference']['net']
    combined_preds = net(inputs['imgs'])
    result = build_targets(combined_preds)
    return result
